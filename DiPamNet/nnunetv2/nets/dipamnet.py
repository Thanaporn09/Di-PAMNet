from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Sequence, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def _distributed_sum_(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def _distributed_gather_cat(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world_size = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=0)

class EMAVectorQuantizer(nn.Module):

    def __init__(
        self,
        num_codes,
        code_dim,
        decay=0.99,
        eps=1e-5,
        beta=0.25,
        usage_reg=True,
        usage_tau=1.0,
        usage_rho=0.10,
        usage_ema_decay=0.99,
        usage_eps=1e-6,
        usage_topm=32,
        use_cosine_sim=True,
        dead_code_threshold=1.0,
        reinit_noise_std=1e-3,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.beta = float(beta)

        self.usage_reg = bool(usage_reg)
        self.usage_tau = float(usage_tau)
        self.usage_rho = float(usage_rho)
        self.usage_ema_decay = float(usage_ema_decay)
        self.usage_eps = float(usage_eps)
        self.usage_topm = int(usage_topm)

        self.use_cosine_sim = bool(use_cosine_sim)
        self.dead_code_threshold = float(dead_code_threshold)
        self.reinit_noise_std = float(reinit_noise_std)

        self.ema_enabled: bool = True

        embed = torch.randn(self.num_codes, self.code_dim, dtype=torch.float32)
        embed = embed / (embed.norm(dim=1, keepdim=True) + 1e-8)

        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_codes, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer(
            "usage_ema",
            torch.full((self.num_codes,), 1.0 / self.num_codes, dtype=torch.float32),
        )
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def initialize_codebook_from_batch(self, z: torch.Tensor):
        if z.numel() == 0:
            return

        z = z.float()
        if self.use_cosine_sim:
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

        z = _distributed_gather_cat(z)
        N = z.shape[0]
        K = self.num_codes

        if N == 0:
            return

        if N >= K:
            idx = torch.randperm(N, device=z.device)[:K]
            sel = z[idx]
        else:
            reps = (K + N - 1) // N
            sel = z.repeat(reps, 1)[:K]
            if self.reinit_noise_std > 0:
                sel = sel + self.reinit_noise_std * torch.randn_like(sel)
                sel = sel / (sel.norm(dim=1, keepdim=True) + 1e-8)

        self.embed.copy_(sel)
        self.embed_avg.copy_(sel)
        self.cluster_size.fill_(1.0)
        self.usage_ema.fill_(1.0 / K)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _reinitialize_dead_codes(self, z_fp32: torch.Tensor):
        dead = self.cluster_size < self.dead_code_threshold
        if not torch.any(dead):
            return

        z_pool = z_fp32
        if self.use_cosine_sim:
            z_pool = z_pool / (z_pool.norm(dim=1, keepdim=True) + 1e-8)

        z_pool = _distributed_gather_cat(z_pool)
        if z_pool.shape[0] == 0:
            return

        dead_idx = torch.where(dead)[0]
        n_dead = int(dead_idx.numel())

        if z_pool.shape[0] >= n_dead:
            pick = torch.randperm(z_pool.shape[0], device=z_pool.device)[:n_dead]
            new_codes = z_pool[pick]
        else:
            reps = (n_dead + z_pool.shape[0] - 1) // z_pool.shape[0]
            new_codes = z_pool.repeat(reps, 1)[:n_dead]

        if self.reinit_noise_std > 0:
            new_codes = new_codes + self.reinit_noise_std * torch.randn_like(new_codes)
            new_codes = new_codes / (new_codes.norm(dim=1, keepdim=True) + 1e-8)

        self.embed[dead_idx] = new_codes
        self.embed_avg[dead_idx] = new_codes
        self.cluster_size[dead_idx] = self.dead_code_threshold + 1.0

    @torch.no_grad()
    def _update_codebook_ema(self, z_fp32: torch.Tensor, onehot_fp32: torch.Tensor):
        cluster_size = onehot_fp32.sum(dim=0)
        embed_sum = onehot_fp32.t() @ z_fp32

        cluster_size = _distributed_sum_(cluster_size)
        embed_sum = _distributed_sum_(embed_sum)

        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
        self.embed.copy_(self.embed / (self.embed.norm(dim=1, keepdim=True) + 1e-8))

        self._reinitialize_dead_codes(z_fp32)

    def _compute_soft_usage_distribution(self, dist_fp32: torch.Tensor):
        tau = max(self.usage_tau, 1e-6)
        K = int(dist_fp32.shape[1])
        M = max(1, min(int(self.usage_topm), K))

        vals, idx = torch.topk(dist_fp32, k=M, dim=1, largest=False, sorted=False)
        a_small = torch.softmax((-vals) / tau, dim=1)

        p_sum = torch.zeros(K, device=dist_fp32.device, dtype=torch.float32)
        p_sum.scatter_add_(0, idx.reshape(-1), a_small.reshape(-1))

        n = torch.tensor([dist_fp32.shape[0]], device=dist_fp32.device, dtype=torch.float32)
        p_sum = _distributed_sum_(p_sum)
        n = _distributed_sum_(n)

        p = p_sum / n.clamp_min(1.0)
        p = p / (p.sum() + self.usage_eps)
        return p

    def _compute_usage_diversity_loss(self, p: torch.Tensor):
        eps = self.usage_eps
        K = self.num_codes
        rho = float(min(max(self.usage_rho, 0.0), 1.0))

        p_ema = self.usage_ema.detach()

        u = torch.full_like(p_ema, 1.0 / K)
        p_mix = (1.0 - rho) * p_ema + rho * u
        p_tgt = 0.9 * p_mix + 0.1 * u
        p_tgt = p_tgt / (p_tgt.sum() + eps)

        div = (p * ((p + eps).log() - (p_tgt + eps).log())).sum()
        ent = -(p * (p + eps).log()).sum()
        return div, ent

    def soft_assign(self, z: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
        assert z.dim() == 2 and z.size(1) == self.code_dim

        z_fp32 = z.float()
        e_fp32 = self.embed.float()

        if self.use_cosine_sim:
            z_assign = z_fp32 / (z_fp32.norm(dim=1, keepdim=True) + 1e-8)
            e_assign = e_fp32 / (e_fp32.norm(dim=1, keepdim=True) + 1e-8)
            logits = z_assign @ e_assign.t()
        else:
            z2 = (z_fp32 ** 2).sum(dim=1, keepdim=True)
            e2 = (e_fp32 ** 2).sum(dim=1).unsqueeze(0)
            ze = z_fp32 @ e_fp32.t()
            dist_mat = z2 + e2 - 2.0 * ze
            logits = -dist_mat

        tau = max(float(tau), 1e-6)
        return torch.softmax(logits / tau, dim=1)

    def forward(self, z: torch.Tensor):
        assert z.dim() == 2 and z.size(1) == self.code_dim

        z_fp32 = z.float()
        z_assign = z_fp32
        if self.use_cosine_sim:
            z_assign = z_assign / (z_assign.norm(dim=1, keepdim=True) + 1e-8)

        if self.training and self.ema_enabled and (not bool(self.initialized.item())):
            self.initialize_codebook_from_batch(z_assign)

        e_fp32 = self.embed.detach().clone()

        z2 = (z_assign ** 2).sum(dim=1, keepdim=True)
        e2 = (e_fp32 ** 2).sum(dim=1).unsqueeze(0)
        ze = z_assign @ e_fp32.t()
        dist_mat = z2 + e2 - 2.0 * ze

        indices = torch.argmin(dist_mat, dim=1)
        onehot = F.one_hot(indices, self.num_codes).float()
        q_fp32 = onehot @ e_fp32

        if self.training and self.ema_enabled:
            self._update_codebook_ema(z_assign, onehot)

        commit_loss = self.beta * F.mse_loss(z_assign, q_fp32.detach())
        commit_per_token = self.beta * ((z_assign - q_fp32.detach()) ** 2).mean(dim=1)

        q = q_fp32.to(dtype=z.dtype)
        if self.use_cosine_sim:
            z_norm = z_assign.to(dtype=z.dtype)
            q_st = z_norm + (q - z_norm).detach()
        else:
            q_st = z + (q - z).detach()

        counts = onehot.sum(dim=0)
        n = torch.tensor([onehot.shape[0]], device=onehot.device, dtype=torch.float32)
        counts = _distributed_sum_(counts)
        n = _distributed_sum_(n)

        avg_probs = counts / n.clamp_min(1.0)
        avg_probs = avg_probs / (avg_probs.sum() + 1e-10)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        if self.usage_reg:
            p = self._compute_soft_usage_distribution(dist_mat)
            if self.training and self.ema_enabled:
                with torch.no_grad():
                    self.usage_ema.mul_(self.usage_ema_decay).add_(
                        p.detach(), alpha=1.0 - self.usage_ema_decay
                    )
                    self.usage_ema.copy_(self.usage_ema / (self.usage_ema.sum() + self.usage_eps))
            div_loss, usage_entropy = self._compute_usage_diversity_loss(p)
        else:
            div_loss = torch.full((), float("nan"), device=z.device, dtype=torch.float32)
            usage_entropy = torch.full((), float("nan"), device=z.device, dtype=torch.float32)

        return q_st, indices, commit_loss, perplexity, div_loss, usage_entropy, commit_per_token

@dataclass
class DiPAMMetrics:
    vq_loss: torch.Tensor

    vq_commit_global: torch.Tensor
    vq_commit_local: torch.Tensor

    vq_div_global: torch.Tensor
    vq_div_local: torch.Tensor

    perplexity_x_global: torch.Tensor
    perplexity_y_global: torch.Tensor
    perplexity_z_global: torch.Tensor

    perplexity_x_local: torch.Tensor
    perplexity_y_local: torch.Tensor
    perplexity_z_local: torch.Tensor

    residual_ratio_global: torch.Tensor
    residual_ratio_local: torch.Tensor
    residual_ratio_total: torch.Tensor

    alpha_mean: torch.Tensor
    beta_mean: torch.Tensor

    usage_entropy_global: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    usage_entropy_local: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

class DiPAM(nn.Module):

    def __init__(
        self,
        in_channels: int,
        stage_idx: int,
        num_stages: int,
        embed_dim: int = 64,
        num_codes_global: int = 128,
        num_codes_local: int = 64,
        code_sep_tau: float = 0.25,
        decay: float = 0.99,
        beta: float = 0.25,
        eps: float = 1e-5,
        usage_reg: bool = True,
        usage_lambda_global: float = 1e-3,
        usage_lambda_local: float = 1e-3,
        usage_tau: float = 1.0,
        usage_rho: float = 0.10,
        usage_ema_decay: float = 0.99,
        usage_topm: int = 32,
        use_layernorm: bool = True,
        pool_mode: Literal["mean", "attn", "topk_attn"] = "topk_attn",
        local_pool_mode: Literal["mean", "attn", "topk_attn"] = "mean",
        attn_temperature: float = 1.0,
        topk_k: int = 8,
        topk_ratio: float | None = None,
        topk_min: int = 4,
        topk_max: int = 12,
        local_grid_x: Tuple[int, int] = (2, 2),
        local_grid_y: Tuple[int, int] = (2, 2),
        local_grid_z: Tuple[int, int] = (2, 2),
        learnable_global_scale: bool = True,
        global_scale_max: float = 1.0,
        global_scale_init: float = 0.5,
        learnable_local_scale: bool = True,
        local_scale_max: float = 1.0,
        local_scale_init: float = 0.5,
        use_axis_softmax_gating: bool = True,
        edge_aware_gate: bool = True,
        edge_k: float = 1.0,
        edge_eps: float = 1e-6,
        gate_detach_edge: bool = True,
        residual_gate: bool = True,
        residual_gate_mode: Literal["voxel", "channel"] = "channel",
        residual_gate_detach_R: bool = True,
        residual_gate_detach_F: bool = False,
        residual_gate_init_bias: float = 2.0,
        ctrl_dim: int = 8,
        tri_dim: Optional[int] = None,
        t_local: Optional[int] = None,
        c_mid_global: Optional[int] = None,
        c_mid_local: Optional[int] = None,
    ):
        super().__init__()
        C = int(in_channels)
        d = int(embed_dim)

        self.in_channels = C
        self.stage_idx = int(stage_idx)
        self.num_stages = int(num_stages)

        self.pool_mode = pool_mode
        self.local_pool_mode = local_pool_mode
        self.attn_temperature = float(attn_temperature)
        self.topk_k = int(topk_k)
        self.topk_ratio = None if topk_ratio is None else float(topk_ratio)
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)
        self.code_sep_tau = float(code_sep_tau)

        self.local_grid_x = (int(local_grid_x[0]), int(local_grid_x[1]))
        self.local_grid_y = (int(local_grid_y[0]), int(local_grid_y[1]))
        self.local_grid_z = (int(local_grid_z[0]), int(local_grid_z[1]))

        self.usage_reg = bool(usage_reg)
        self.usage_lambda_global = float(usage_lambda_global)
        self.usage_lambda_local = float(usage_lambda_local)

        self.use_axis_softmax_gating = bool(use_axis_softmax_gating)
        self.edge_aware_gate = bool(edge_aware_gate)
        self.edge_k = float(edge_k)
        self.edge_eps = float(edge_eps)
        self.gate_detach_edge = bool(gate_detach_edge)

        self.learnable_global_scale = bool(learnable_global_scale)
        self.global_scale_max = float(global_scale_max)

        self.learnable_local_scale = bool(learnable_local_scale)
        self.local_scale_max = float(local_scale_max)

        self.residual_gate = bool(residual_gate)
        self.residual_gate_mode = residual_gate_mode
        self.residual_gate_detach_R = bool(residual_gate_detach_R)
        self.residual_gate_detach_F = bool(residual_gate_detach_F)

        tri_dim = int(tri_dim) if tri_dim is not None else min(16, max(8, C // 16))
        t_local = int(t_local) if t_local is not None else min(16, max(8, C // 16))
        c_mid_global = int(c_mid_global) if c_mid_global is not None else min(24, max(16, C // 12))
        c_mid_local = int(c_mid_local) if c_mid_local is not None else min(24, max(16, C // 12))

        self.tri_dim = tri_dim
        self.t_local = t_local
        self.c_mid_global = c_mid_global
        self.c_mid_local = c_mid_local

        self.attn_x = nn.Conv3d(C, 1, kernel_size=1, bias=True)
        self.attn_y = nn.Conv3d(C, 1, kernel_size=1, bias=True)
        self.attn_z = nn.Conv3d(C, 1, kernel_size=1, bias=True)

        self.proj_x = nn.Linear(C, d, bias=True)
        self.proj_y = nn.Linear(C, d, bias=True)
        self.proj_z = nn.Linear(C, d, bias=True)

        self.norm_x = nn.LayerNorm(d) if use_layernorm else nn.Identity()
        self.norm_y = nn.LayerNorm(d) if use_layernorm else nn.Identity()
        self.norm_z = nn.LayerNorm(d) if use_layernorm else nn.Identity()

        vq_kwargs = dict(
            decay=decay,
            eps=eps,
            beta=beta,
            usage_reg=self.usage_reg,
            usage_tau=usage_tau,
            usage_rho=usage_rho,
            usage_ema_decay=usage_ema_decay,
            usage_topm=usage_topm,
        )

        self.vq_xg = EMAVectorQuantizer(num_codes_global, d, **vq_kwargs)
        self.vq_yg = EMAVectorQuantizer(num_codes_global, d, **vq_kwargs)
        self.vq_zg = EMAVectorQuantizer(num_codes_global, d, **vq_kwargs)

        self.vq_xl = EMAVectorQuantizer(num_codes_local, d, **vq_kwargs)
        self.vq_yl = EMAVectorQuantizer(num_codes_local, d, **vq_kwargs)
        self.vq_zl = EMAVectorQuantizer(num_codes_local, d, **vq_kwargs)

        self.tri_x = nn.Linear(d, tri_dim, bias=True)
        self.tri_y = nn.Linear(d, tri_dim, bias=True)
        self.tri_z = nn.Linear(d, tri_dim, bias=True)

        self.global_fuse = nn.Sequential(
            nn.Conv3d(3 * tri_dim, c_mid_global, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(
                c_mid_global,
                c_mid_global,
                kernel_size=3,
                padding=1,
                groups=c_mid_global,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv3d(c_mid_global, C, kernel_size=1, bias=True),
        )

        self.local_to_t_x = nn.Linear(d, t_local, bias=True)
        self.local_to_t_y = nn.Linear(d, t_local, bias=True)
        self.local_to_t_z = nn.Linear(d, t_local, bias=True)

        self.local_refine_2d = nn.Sequential(
            nn.Conv2d(t_local, t_local, kernel_size=3, padding=1, groups=t_local, bias=False),
            nn.GELU(),
            nn.Conv2d(t_local, t_local, kernel_size=1, bias=True),
        )

        self.local_fuse = nn.Sequential(
            nn.Conv3d(3 * t_local, c_mid_local, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(
                c_mid_local,
                c_mid_local,
                kernel_size=3,
                padding=1,
                groups=c_mid_local,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv3d(c_mid_local, C, kernel_size=1, bias=True),
        )

        self.global_axis_gate = nn.Conv3d(C, 3, kernel_size=1, bias=True)
        self.local_axis_gate = nn.Conv3d(C, 3, kernel_size=1, bias=True)

        self.stage_embed = nn.Embedding(self.num_stages, int(ctrl_dim))
        self.ctrl_mlp = nn.Sequential(
            nn.Linear(C + int(ctrl_dim), 16, bias=True),
            nn.GELU(),
            nn.Linear(16, 3, bias=True),
        )

        g_init = max(1e-6, min(float(global_scale_init), self.global_scale_max))
        g_p = min(max(g_init / self.global_scale_max, 1e-6), 1.0 - 1e-6)
        g_raw = torch.tensor(float(torch.log(torch.tensor(g_p / (1.0 - g_p))).item()), dtype=torch.float32)
        if self.learnable_global_scale:
            self.global_scale_raw = nn.Parameter(g_raw)
        else:
            self.register_buffer("global_scale_raw", g_raw)

        l_init = max(1e-6, min(float(local_scale_init), self.local_scale_max))
        l_p = min(max(l_init / self.local_scale_max, 1e-6), 1.0 - 1e-6)
        l_raw = torch.tensor(float(torch.log(torch.tensor(l_p / (1.0 - l_p))).item()), dtype=torch.float32)
        if self.learnable_local_scale:
            self.local_scale_raw = nn.Parameter(l_raw)
        else:
            self.register_buffer("local_scale_raw", l_raw)

        if self.residual_gate:
            out_ch = 1 if residual_gate_mode == "voxel" else C
            self.res_gate = nn.Sequential(
                nn.Conv3d(C * 2, C, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv3d(C, out_ch, kernel_size=1, bias=True),
            )
            nn.init.constant_(self.res_gate[-1].bias, float(residual_gate_init_bias))

        self.last_aux: Dict[str, Any] = {}

    def _clear_auxiliary_memory_outputs(self):
        self.last_aux = {}

    def _store_auxiliary_memory_outputs(
        self,
        F_in: torch.Tensor,
        Gx: int,
        Gy: int,
        Gz: int,
        idx_xg: torch.Tensor,
        idx_yg: torch.Tensor,
        idx_zg: torch.Tensor,
        cpt_xg: torch.Tensor,
        cpt_yg: torch.Tensor,
        cpt_zg: torch.Tensor,
        idx_xl: torch.Tensor,
        idx_yl: torch.Tensor,
        idx_zl: torch.Tensor,
        cpt_xl: torch.Tensor,
        cpt_yl: torch.Tensor,
        cpt_zl: torch.Tensor,
        soft_xl: torch.Tensor,
        soft_yl: torch.Tensor,
        soft_zl: torch.Tensor,
    ):
        B, C, D, H, W = F_in.shape
        Kx = int(self.vq_xl.num_codes)
        Ky = int(self.vq_yl.num_codes)
        Kz = int(self.vq_zl.num_codes)

        self.last_aux = {
            "shape": (B, C, D, H, W),
            "num_codes_global": int(self.vq_xg.num_codes),
            "num_codes_local": int(self.vq_xl.num_codes),
            "global": {
                "x": {
                    "indices": idx_xg.view(B, W),
                    "commit": cpt_xg.view(B, W),
                },
                "y": {
                    "indices": idx_yg.view(B, H),
                    "commit": cpt_yg.view(B, H),
                },
                "z": {
                    "indices": idx_zg.view(B, D),
                    "commit": cpt_zg.view(B, D),
                },
            },
            "local": {
                "x": {
                    "indices": idx_xl.view(B, W, Gx),
                    "commit": cpt_xl.view(B, W, Gx),
                    "soft_assign": soft_xl.view(B, W, Gx, Kx),
                    "grid": self.local_grid_x,
                },
                "y": {
                    "indices": idx_yl.view(B, H, Gy),
                    "commit": cpt_yl.view(B, H, Gy),
                    "soft_assign": soft_yl.view(B, H, Gy, Ky),
                    "grid": self.local_grid_y,
                },
                "z": {
                    "indices": idx_zl.view(B, D, Gz),
                    "commit": cpt_zl.view(B, D, Gz),
                    "soft_assign": soft_zl.view(B, D, Gz, Kz),
                    "grid": self.local_grid_z,
                },
            },
        }

    def _get_bounded_scale(self, raw, max_val: float, dtype, device):
        val = float(max_val) * torch.sigmoid(raw.to(device=device))
        return val.to(dtype=dtype)

    def _compute_k(self, L: int) -> int:
        if self.topk_ratio is None:
            return max(1, min(self.topk_k, int(L)))
        k = int(round(self.topk_ratio * L))
        k = max(self.topk_min, min(k, self.topk_max))
        return max(1, min(k, int(L)))

    def _compute_residual_memory_gate(
        self,
        F_in: torch.Tensor,
        R: torch.Tensor,
        gate_bias: Optional[torch.Tensor] = None,
    ):
        Fin = F_in.detach() if self.residual_gate_detach_F else F_in
        Rin = R.detach() if self.residual_gate_detach_R else R
        logits = self.res_gate(torch.cat([Fin, Rin], dim=1))
        if gate_bias is not None:
            logits = logits + gate_bias
        g = torch.sigmoid(logits)
        if self.residual_gate_mode == "voxel":
            g = g.expand_as(R)
        return g

    def _topk_attention_pool_1d(self, feats: torch.Tensor, logits: torch.Tensor, k: int, tau: float) -> torch.Tensor:
        N, C, L = feats.shape
        k_eff = max(1, min(int(k), int(L)))
        if k_eff >= L:
            w = torch.softmax(logits.float() / tau, dim=1).to(dtype=feats.dtype)
            return (feats * w.unsqueeze(1)).sum(dim=2)
        vals, idx = torch.topk(logits.float(), k_eff, dim=1, largest=True, sorted=False)
        w = torch.softmax(vals / tau, dim=1).to(dtype=feats.dtype)
        feats_k = torch.gather(feats, dim=2, index=idx.unsqueeze(1).expand(N, C, k_eff))
        return (feats_k * w.unsqueeze(1)).sum(dim=2)

    def _extract_global_tokens_mean(self, F_in: torch.Tensor):
        Tx = F_in.mean(dim=(2, 3)).permute(0, 2, 1).contiguous()
        Ty = F_in.mean(dim=(2, 4)).permute(0, 2, 1).contiguous()
        Tz = F_in.mean(dim=(3, 4)).permute(0, 2, 1).contiguous()
        return Tx, Ty, Tz

    def _extract_global_tokens_attention(self, F_in: torch.Tensor):
        B, C, D, H, W = F_in.shape
        tau = max(self.attn_temperature, 1e-6)

        lx = self.attn_x(F_in).squeeze(1).permute(0, 3, 1, 2).contiguous().view(B * W, D * H).float()
        ax = torch.softmax(lx / tau, dim=1).to(dtype=F_in.dtype)
        Fx = F_in.permute(0, 4, 1, 2, 3).contiguous().view(B * W, C, D * H)
        Tx = (Fx * ax.unsqueeze(1)).sum(dim=2).view(B, W, C)

        ly = self.attn_y(F_in).squeeze(1).permute(0, 2, 1, 3).contiguous().view(B * H, D * W).float()
        ay = torch.softmax(ly / tau, dim=1).to(dtype=F_in.dtype)
        Fy = F_in.permute(0, 3, 1, 2, 4).contiguous().view(B * H, C, D * W)
        Ty = (Fy * ay.unsqueeze(1)).sum(dim=2).view(B, H, C)

        lz = self.attn_z(F_in).squeeze(1).contiguous().view(B * D, H * W).float()
        az = torch.softmax(lz / tau, dim=1).to(dtype=F_in.dtype)
        Fz = F_in.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H * W)
        Tz = (Fz * az.unsqueeze(1)).sum(dim=2).view(B, D, C)

        return Tx, Ty, Tz

    def _extract_global_tokens_topk_attention(self, F_in: torch.Tensor):
        B, C, D, H, W = F_in.shape
        tau = max(self.attn_temperature, 1e-6)

        lx = self.attn_x(F_in).squeeze(1).permute(0, 3, 1, 2).contiguous().view(B * W, D * H)
        Fx = F_in.permute(0, 4, 1, 2, 3).contiguous().view(B * W, C, D * H)
        Tx = self._topk_attention_pool_1d(Fx, lx, self._compute_k(D * H), tau).view(B, W, C)

        ly = self.attn_y(F_in).squeeze(1).permute(0, 2, 1, 3).contiguous().view(B * H, D * W)
        Fy = F_in.permute(0, 3, 1, 2, 4).contiguous().view(B * H, C, D * W)
        Ty = self._topk_attention_pool_1d(Fy, ly, self._compute_k(D * W), tau).view(B, H, C)

        lz = self.attn_z(F_in).squeeze(1).contiguous().view(B * D, H * W)
        Fz = F_in.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H * W)
        Tz = self._topk_attention_pool_1d(Fz, lz, self._compute_k(H * W), tau).view(B, D, C)

        return Tx, Ty, Tz

    @staticmethod
    def _pad_plane_to_local_grid(x: torch.Tensor, grid_p: int, grid_q: int):
        P, Q = x.shape[-2], x.shape[-1]
        pad_p = (grid_p - (P % grid_p)) % grid_p
        pad_q = (grid_q - (Q % grid_q)) % grid_q
        if pad_p or pad_q:
            x = F.pad(x, (0, pad_q, 0, pad_p))
        return x

    def _extract_local_plane_tokens(
        self,
        feats2d: torch.Tensor,
        logits2d: torch.Tensor,
        grid_p: int,
        grid_q: int,
        mode: Literal["mean", "attn", "topk_attn"],
    ) -> torch.Tensor:
        assert feats2d.dim() == 4 and logits2d.dim() == 4
        N, C, P0, Q0 = feats2d.shape

        grid_p = max(1, int(grid_p))
        grid_q = max(1, int(grid_q))

        feats2d = self._pad_plane_to_local_grid(feats2d, grid_p, grid_q)
        logits2d = self._pad_plane_to_local_grid(logits2d, grid_p, grid_q)

        Pp, Qp = feats2d.shape[-2], feats2d.shape[-1]
        win_p = Pp // grid_p
        win_q = Qp // grid_q
        G = grid_p * grid_q
        L = win_p * win_q
        tau = max(self.attn_temperature, 1e-6)

        if mode == "mean":
            pooled = F.adaptive_avg_pool2d(feats2d, output_size=(grid_p, grid_q))
            tok = pooled.flatten(2).transpose(1, 2).contiguous()
            return tok

        patches = F.unfold(feats2d, kernel_size=(win_p, win_q), stride=(win_p, win_q))
        patches = patches.view(N, C, L, G)

        log_patches = F.unfold(logits2d, kernel_size=(win_p, win_q), stride=(win_p, win_q)).view(N, L, G)

        if mode == "attn":
            w = torch.softmax(log_patches.float() / tau, dim=1).to(dtype=feats2d.dtype)
            tok = (patches * w.unsqueeze(1)).sum(dim=2).permute(0, 2, 1).contiguous()
            return tok

        if mode == "topk_attn":
            k = self._compute_k(L)
            vals, idx = torch.topk(log_patches.float(), k, dim=1, largest=True, sorted=False)
            w = torch.softmax(vals / tau, dim=1).to(dtype=feats2d.dtype)
            idx_exp = idx.unsqueeze(1).expand(N, C, k, G)
            patches_k = torch.gather(patches, dim=2, index=idx_exp)
            tok = (patches_k * w.unsqueeze(1)).sum(dim=2).permute(0, 2, 1).contiguous()
            return tok

        raise ValueError(f"Unknown local mode={mode}")

    @staticmethod
    def _compute_axis_edge_proxy(F_in: torch.Tensor):
        dz = F.pad(F_in[:, :, 1:] - F_in[:, :, :-1], (0, 0, 0, 0, 0, 1))
        dy = F.pad(F_in[:, :, :, 1:] - F_in[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
        dx = F.pad(F_in[:, :, :, :, 1:] - F_in[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))

        def _norm(t):
            m = t.pow(2).mean(dim=1, keepdim=True).sqrt()
            B = m.shape[0]
            denom = m.flatten(1).mean(dim=1, keepdim=True).clamp_min(1e-6)
            return m / denom.view(B, 1, 1, 1, 1)

        return _norm(dx), _norm(dy), _norm(dz)

    def _compute_global_axis_gate(self, F_in: torch.Tensor):
        B, _, D, H, W = F_in.shape
        if not self.use_axis_softmax_gating:
            return torch.ones(B, 3, D, H, W, device=F_in.device, dtype=F_in.dtype)

        g0 = torch.softmax(self.global_axis_gate(F_in), dim=1)
        if not self.edge_aware_gate:
            return g0

        ex, ey, ez = self._compute_axis_edge_proxy(F_in)
        if self.gate_detach_edge:
            ex, ey, ez = ex.detach(), ey.detach(), ez.detach()

        sup = torch.cat(
            [
                torch.exp(-self.edge_k * ex),
                torch.exp(-self.edge_k * ey),
                torch.exp(-self.edge_k * ez),
            ],
            dim=1,
        ).clamp_min(self.edge_eps)

        g = g0 * sup
        g = g / (g.sum(dim=1, keepdim=True) + self.edge_eps)
        return g

    def _compute_local_axis_gate(self, F_in: torch.Tensor):
        B, _, D, H, W = F_in.shape
        if not self.use_axis_softmax_gating:
            return torch.ones(B, 3, D, H, W, device=F_in.device, dtype=F_in.dtype)

        g0 = torch.softmax(self.local_axis_gate(F_in), dim=1)
        if not self.edge_aware_gate:
            return g0

        ex, ey, ez = self._compute_axis_edge_proxy(F_in)
        if self.gate_detach_edge:
            ex, ey, ez = ex.detach(), ey.detach(), ez.detach()

        enh = torch.cat(
            [
                1.0 + self.edge_k * ex,
                1.0 + self.edge_k * ey,
                1.0 + self.edge_k * ez,
            ],
            dim=1,
        )

        g = g0 * enh
        g = g / (g.sum(dim=1, keepdim=True) + self.edge_eps)
        return g

    def _reconstruct_local_plane(self, tok_t: torch.Tensor, grid_p: int, grid_q: int, out_p: int, out_q: int):
        N, G, T = tok_t.shape
        assert G == grid_p * grid_q, f"G={G}, but grid={grid_p}x{grid_q}"

        x = tok_t.view(N, grid_p, grid_q, T).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(out_p, out_q), mode="bilinear", align_corners=False)
        x = self.local_refine_2d(x)
        return x

    def _extract_local_axis_tokens(self, F_in: torch.Tensor):
        B, C, D, H, W = F_in.shape

        feats_x = F_in.permute(0, 4, 1, 2, 3).contiguous().view(B * W, C, D, H)
        logit_x = self.attn_x(F_in).permute(0, 4, 1, 2, 3).contiguous().view(B * W, 1, D, H)
        TxL = self._extract_local_plane_tokens(feats_x, logit_x, self.local_grid_x[0], self.local_grid_x[1], self.local_pool_mode)
        Gx = TxL.shape[1]
        TxL = TxL.view(B, W, Gx, C).reshape(B, W * Gx, C)

        feats_y = F_in.permute(0, 3, 1, 2, 4).contiguous().view(B * H, C, D, W)
        logit_y = self.attn_y(F_in).permute(0, 3, 1, 2, 4).contiguous().view(B * H, 1, D, W)
        TyL = self._extract_local_plane_tokens(feats_y, logit_y, self.local_grid_y[0], self.local_grid_y[1], self.local_pool_mode)
        Gy = TyL.shape[1]
        TyL = TyL.view(B, H, Gy, C).reshape(B, H * Gy, C)

        feats_z = F_in.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        logit_z = self.attn_z(F_in).permute(0, 2, 1, 3, 4).contiguous().view(B * D, 1, H, W)
        TzL = self._extract_local_plane_tokens(feats_z, logit_z, self.local_grid_z[0], self.local_grid_z[1], self.local_pool_mode)
        Gz = TzL.shape[1]
        TzL = TzL.view(B, D, Gz, C).reshape(B, D * Gz, C)

        return TxL, Gx, TyL, Gy, TzL, Gz

    def _reconstruct_global_memory(self, qxg: torch.Tensor, qyg: torch.Tensor, qzg: torch.Tensor, F_in: torch.Tensor):
        B, C, D, H, W = F_in.shape

        Tx = self.tri_x(qxg)
        Ty = self.tri_y(qyg)
        Tz = self.tri_z(qzg)

        X = Tx.permute(0, 2, 1).contiguous().view(B, self.tri_dim, 1, 1, W).expand(B, self.tri_dim, D, H, W)
        Y = Ty.permute(0, 2, 1).contiguous().view(B, self.tri_dim, 1, H, 1).expand(B, self.tri_dim, D, H, W)
        Z = Tz.permute(0, 2, 1).contiguous().view(B, self.tri_dim, D, 1, 1).expand(B, self.tri_dim, D, H, W)

        g = self._compute_global_axis_gate(F_in)
        X = X * g[:, 0:1]
        Y = Y * g[:, 1:2]
        Z = Z * g[:, 2:3]

        return self.global_fuse(torch.cat([X, Y, Z], dim=1))

    def _reconstruct_local_memory(
        self,
        qxL: torch.Tensor,
        qyL: torch.Tensor,
        qzL: torch.Tensor,
        F_in: torch.Tensor,
        Gx: int,
        Gy: int,
        Gz: int,
    ):
        B, C, D, H, W = F_in.shape

        tx = self.local_to_t_x(qxL.view(B, W, Gx, -1)).view(B * W, Gx, self.t_local)
        Rx2d = self._reconstruct_local_plane(tx, self.local_grid_x[0], self.local_grid_x[1], D, H)
        Rx = Rx2d.view(B, W, self.t_local, D, H).permute(0, 2, 3, 4, 1).contiguous()

        ty = self.local_to_t_y(qyL.view(B, H, Gy, -1)).view(B * H, Gy, self.t_local)
        Ry2d = self._reconstruct_local_plane(ty, self.local_grid_y[0], self.local_grid_y[1], D, W)
        Ry = Ry2d.view(B, H, self.t_local, D, W).permute(0, 2, 3, 1, 4).contiguous()

        tz = self.local_to_t_z(qzL.view(B, D, Gz, -1)).view(B * D, Gz, self.t_local)
        Rz2d = self._reconstruct_local_plane(tz, self.local_grid_z[0], self.local_grid_z[1], H, W)
        Rz = Rz2d.view(B, D, self.t_local, H, W).permute(0, 2, 1, 3, 4).contiguous()

        g = self._compute_local_axis_gate(F_in)
        Rx = Rx * g[:, 0:1]
        Ry = Ry * g[:, 1:2]
        Rz = Rz * g[:, 2:3]

        return self.local_fuse(torch.cat([Rx, Ry, Rz], dim=1))

    def _compute_stage_adaptive_mixing(self, F_in: torch.Tensor):
        B, C, _, _, _ = F_in.shape
        gap = F_in.mean(dim=(2, 3, 4))
        sid = torch.full((B,), self.stage_idx, dtype=torch.long, device=F_in.device)
        se = self.stage_embed(sid)
        ctrl = self.ctrl_mlp(torch.cat([gap, se], dim=1))

        base_g = self._get_bounded_scale(self.global_scale_raw, self.global_scale_max, F_in.dtype, F_in.device)
        base_l = self._get_bounded_scale(self.local_scale_raw, self.local_scale_max, F_in.dtype, F_in.device)

        alpha = (base_g * torch.sigmoid(ctrl[:, 0])).view(B, 1, 1, 1, 1)
        beta = (base_l * torch.sigmoid(ctrl[:, 1])).view(B, 1, 1, 1, 1)
        gate_bias = ctrl[:, 2].view(B, 1, 1, 1, 1)
        return alpha, beta, gate_bias

    def forward(self, F_in: torch.Tensor):
        assert F_in.dim() == 5, f"Expected [B,C,D,H,W], got {tuple(F_in.shape)}"
        self._clear_auxiliary_memory_outputs()

        if self.pool_mode == "mean":
            Txg, Tyg, Tzg = self._extract_global_tokens_mean(F_in)
        elif self.pool_mode == "attn":
            Txg, Tyg, Tzg = self._extract_global_tokens_attention(F_in)
        elif self.pool_mode == "topk_attn":
            Txg, Tyg, Tzg = self._extract_global_tokens_topk_attention(F_in)
        else:
            raise ValueError(f"Unknown pool_mode={self.pool_mode}")

        Uxg = self.norm_x(self.proj_x(Txg))
        Uyg = self.norm_y(self.proj_y(Tyg))
        Uzg = self.norm_z(self.proj_z(Tzg))

        qxg, idx_xg, loss_xg, ppl_xg, div_xg, ent_xg, cpt_xg = self.vq_xg(Uxg.reshape(-1, Uxg.shape[-1]))
        qyg, idx_yg, loss_yg, ppl_yg, div_yg, ent_yg, cpt_yg = self.vq_yg(Uyg.reshape(-1, Uyg.shape[-1]))
        qzg, idx_zg, loss_zg, ppl_zg, div_zg, ent_zg, cpt_zg = self.vq_zg(Uzg.reshape(-1, Uzg.shape[-1]))

        qxg = qxg.view_as(Uxg)
        qyg = qyg.view_as(Uyg)
        qzg = qzg.view_as(Uzg)

        R_global = self._reconstruct_global_memory(qxg, qyg, qzg, F_in)

        TxL, Gx, TyL, Gy, TzL, Gz = self._extract_local_axis_tokens(F_in)

        UxL = self.norm_x(self.proj_x(TxL))
        UyL = self.norm_y(self.proj_y(TyL))
        UzL = self.norm_z(self.proj_z(TzL))

        qxL, idx_xl, loss_xl, ppl_xl, div_xl, ent_xl, cpt_xl = self.vq_xl(UxL.reshape(-1, UxL.shape[-1]))
        qyL, idx_yl, loss_yl, ppl_yl, div_yl, ent_yl, cpt_yl = self.vq_yl(UyL.reshape(-1, UyL.shape[-1]))
        qzL, idx_zl, loss_zl, ppl_zl, div_zl, ent_zl, cpt_zl = self.vq_zl(UzL.reshape(-1, UzL.shape[-1]))

        soft_xl = self.vq_xl.soft_assign(UxL.reshape(-1, UxL.shape[-1]), tau=self.code_sep_tau)
        soft_yl = self.vq_yl.soft_assign(UyL.reshape(-1, UyL.shape[-1]), tau=self.code_sep_tau)
        soft_zl = self.vq_zl.soft_assign(UzL.reshape(-1, UzL.shape[-1]), tau=self.code_sep_tau)

        qxL = qxL.view_as(UxL)
        qyL = qyL.view_as(UyL)
        qzL = qzL.view_as(UzL)

        R_local = self._reconstruct_local_memory(qxL, qyL, qzL, F_in, Gx, Gy, Gz)

        self._store_auxiliary_memory_outputs(
            F_in=F_in,
            Gx=Gx,
            Gy=Gy,
            Gz=Gz,
            idx_xg=idx_xg,
            idx_yg=idx_yg,
            idx_zg=idx_zg,
            cpt_xg=cpt_xg,
            cpt_yg=cpt_yg,
            cpt_zg=cpt_zg,
            idx_xl=idx_xl,
            idx_yl=idx_yl,
            idx_zl=idx_zl,
            cpt_xl=cpt_xl,
            cpt_yl=cpt_yl,
            cpt_zl=cpt_zl,
            soft_xl=soft_xl,
            soft_yl=soft_yl,
            soft_zl=soft_zl,
        )

        alpha, beta, gate_bias = self._compute_stage_adaptive_mixing(F_in)
        R = alpha * R_global + beta * R_local

        if self.residual_gate:
            R = R * self._compute_residual_memory_gate(F_in, R, gate_bias=gate_bias)

        F_out = F_in + R

        vq_commit_global = loss_xg + loss_yg + loss_zg
        vq_commit_local = loss_xl + loss_yl + loss_zl

        vq_div_global = (div_xg + div_yg + div_zg) if self.usage_reg else F_in.new_zeros(())
        vq_div_local = (div_xl + div_yl + div_zl) if self.usage_reg else F_in.new_zeros(())

        vq_loss = (
            vq_commit_global
            + vq_commit_local
            + self.usage_lambda_global * vq_div_global
            + self.usage_lambda_local * vq_div_local
        )

        rr_g = ((R_global.flatten(1).norm(dim=1)) / (F_in.flatten(1).norm(dim=1) + 1e-6)).mean().detach()
        rr_l = ((R_local.flatten(1).norm(dim=1)) / (F_in.flatten(1).norm(dim=1) + 1e-6)).mean().detach()
        rr_t = ((R.flatten(1).norm(dim=1)) / (F_in.flatten(1).norm(dim=1) + 1e-6)).mean().detach()

        metrics = DiPAMMetrics(
            vq_loss=vq_loss,
            vq_commit_global=vq_commit_global,
            vq_commit_local=vq_commit_local,
            vq_div_global=vq_div_global,
            vq_div_local=vq_div_local,
            perplexity_x_global=ppl_xg.detach(),
            perplexity_y_global=ppl_yg.detach(),
            perplexity_z_global=ppl_zg.detach(),
            perplexity_x_local=ppl_xl.detach(),
            perplexity_y_local=ppl_yl.detach(),
            perplexity_z_local=ppl_zl.detach(),
            residual_ratio_global=rr_g,
            residual_ratio_local=rr_l,
            residual_ratio_total=rr_t,
            alpha_mean=alpha.mean().detach(),
            beta_mean=beta.mean().detach(),
            usage_entropy_global=((ent_xg + ent_yg + ent_zg) / 3.0).detach(),
            usage_entropy_local=((ent_xl + ent_yl + ent_zl) / 3.0).detach(),
        )
        return F_out, metrics

class DiPAMNet(nn.Module):

    def __init__(
        self,
        base_net: nn.Module,
        enc_channels: Sequence[int],
        plugin_stage_indices: Optional[Sequence[int]] = None,
        plugin_stage_mask: Optional[Sequence[bool]] = None,
        dipam_embed_dim: int = 64,
        dipam_num_codes_global: int = 128,
        dipam_num_codes_local: int = 64,
        dipam_code_sep_tau: float = 0.25,
        dipam_decay: float = 0.99,
        dipam_beta: float = 0.25,
        dipam_usage_reg: bool = True,
        dipam_usage_lambda_global: float = 1e-3,
        dipam_usage_lambda_local: float = 1e-3,
        dipam_usage_tau: float = 1.0,
        dipam_usage_rho: float = 0.10,
        dipam_usage_ema_decay: float = 0.99,
        dipam_usage_topm: int = 32,
        dipam_global_pool_mode: Literal["mean", "attn", "topk_attn"] = "topk_attn",
        dipam_local_pool_mode: Literal["mean", "attn", "topk_attn"] = "mean",
        dipam_attn_temperature: float = 1.0,
        dipam_topk_k: int = 8,
        dipam_topk_ratio: float | None = None,
        dipam_topk_min: int = 4,
        dipam_topk_max: int = 12,
        dipam_local_grid_x: Tuple[int, int] = (2, 2),
        dipam_local_grid_y: Tuple[int, int] = (2, 2),
        dipam_local_grid_z: Tuple[int, int] = (2, 2),
        dipam_learnable_global_scale: bool = True,
        dipam_global_scale_max: float = 1.0,
        dipam_global_scale_init: float = 0.5,
        dipam_learnable_local_scale: bool = True,
        dipam_local_scale_max: float = 1.0,
        dipam_local_scale_init: float = 0.5,
        dipam_use_axis_softmax_gating: bool = True,
        dipam_edge_aware_gate: bool = True,
        dipam_edge_k: float = 1.0,
        dipam_gate_detach_edge: bool = True,
        dipam_residual_gate: bool = True,
        dipam_residual_gate_mode: Literal["voxel", "channel"] = "channel",
        dipam_residual_gate_detach_R: bool = True,
        dipam_residual_gate_detach_F: bool = False,
        dipam_residual_gate_init_bias: float = 2.0,
        dipam_ctrl_dim: int = 8,
        dipam_tri_dim: Optional[int] = None,
        dipam_t_local: Optional[int] = None,
        dipam_c_mid_global: Optional[int] = None,
        dipam_c_mid_local: Optional[int] = None,
    ):
        super().__init__()
        self.base = base_net
        self.enc_channels = list(map(int, enc_channels))

        enc_attr, dec_attr = self._resolve_enc_dec(self.base)
        self.encoder = getattr(self.base, enc_attr)
        self.decoder = getattr(self.base, dec_attr)

        self.num_stages = len(self.enc_channels)

        self.active_stage_indices = self._resolve_active_stage_indices(
            num_stages=self.num_stages,
            plugin_stage_indices=plugin_stage_indices,
            plugin_stage_mask=plugin_stage_mask,
        )

        self._dipam_cfg = dict(
            embed_dim=int(dipam_embed_dim),
            num_codes_global=int(dipam_num_codes_global),
            num_codes_local=int(dipam_num_codes_local),
            code_sep_tau=float(dipam_code_sep_tau),
            decay=float(dipam_decay),
            beta=float(dipam_beta),
            usage_reg=bool(dipam_usage_reg),
            usage_lambda_global=float(dipam_usage_lambda_global),
            usage_lambda_local=float(dipam_usage_lambda_local),
            usage_tau=float(dipam_usage_tau),
            usage_rho=float(dipam_usage_rho),
            usage_ema_decay=float(dipam_usage_ema_decay),
            usage_topm=int(dipam_usage_topm),
            pool_mode=dipam_global_pool_mode,
            local_pool_mode=dipam_local_pool_mode,
            attn_temperature=float(dipam_attn_temperature),
            topk_k=int(dipam_topk_k),
            topk_ratio=None if dipam_topk_ratio is None else float(dipam_topk_ratio),
            topk_min=int(dipam_topk_min),
            topk_max=int(dipam_topk_max),
            local_grid_x=tuple(dipam_local_grid_x),
            local_grid_y=tuple(dipam_local_grid_y),
            local_grid_z=tuple(dipam_local_grid_z),
            learnable_global_scale=bool(dipam_learnable_global_scale),
            global_scale_max=float(dipam_global_scale_max),
            global_scale_init=float(dipam_global_scale_init),
            learnable_local_scale=bool(dipam_learnable_local_scale),
            local_scale_max=float(dipam_local_scale_max),
            local_scale_init=float(dipam_local_scale_init),
            use_axis_softmax_gating=bool(dipam_use_axis_softmax_gating),
            edge_aware_gate=bool(dipam_edge_aware_gate),
            edge_k=float(dipam_edge_k),
            gate_detach_edge=bool(dipam_gate_detach_edge),
            residual_gate=bool(dipam_residual_gate),
            residual_gate_mode=dipam_residual_gate_mode,
            residual_gate_detach_R=bool(dipam_residual_gate_detach_R),
            residual_gate_detach_F=bool(dipam_residual_gate_detach_F),
            residual_gate_init_bias=float(dipam_residual_gate_init_bias),
            ctrl_dim=int(dipam_ctrl_dim),
            tri_dim=dipam_tri_dim,
            t_local=dipam_t_local,
            c_mid_global=dipam_c_mid_global,
            c_mid_local=dipam_c_mid_local,
        )

        self.dipam_stages = nn.ModuleDict()
        for i in self.active_stage_indices:
            self.dipam_stages[str(i)] = DiPAM(
                in_channels=int(self.enc_channels[i]),
                stage_idx=i,
                num_stages=self.num_stages,
                **self._dipam_cfg,
            )

        self.dipam_last_metrics: Dict[str, DiPAMMetrics] = {}
        self.dipam_last_aux: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _resolve_enc_dec(base: nn.Module) -> Tuple[str, str]:
        if hasattr(base, "encoder") and hasattr(base, "decoder"):
            return "encoder", "decoder"
        if hasattr(base, "conv_encoder") and hasattr(base, "conv_decoder"):
            return "conv_encoder", "conv_decoder"
        raise RuntimeError(
            f"base_net must have encoder/decoder or conv_encoder/conv_decoder. Got: {type(base)}"
        )

    @staticmethod
    def _resolve_active_stage_indices(
        num_stages: int,
        plugin_stage_indices: Optional[Sequence[int]],
        plugin_stage_mask: Optional[Sequence[bool]],
    ) -> List[int]:
        if plugin_stage_indices is not None and plugin_stage_mask is not None:
            raise ValueError("Specify only one of plugin_stage_indices or plugin_stage_mask, not both.")

        if plugin_stage_mask is not None:
            if len(plugin_stage_mask) != num_stages:
                raise ValueError(
                    f"plugin_stage_mask length must equal num_stages={num_stages}, got {len(plugin_stage_mask)}"
                )
            out = [i for i, flag in enumerate(plugin_stage_mask) if bool(flag)]
            return out

        if plugin_stage_indices is None:
            return list(range(num_stages))

        out = []
        for idx in plugin_stage_indices:
            i = int(idx)
            if i < 0:
                i = num_stages + i
            if i < 0 or i >= num_stages:
                raise ValueError(f"Invalid stage index {idx} for num_stages={num_stages}")
            out.append(i)

        out = sorted(set(out))
        return out

    def set_active_stage_indices(self, plugin_stage_indices: Sequence[int]):
        self.active_stage_indices = self._resolve_active_stage_indices(
            self.num_stages, plugin_stage_indices, None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if not isinstance(feats, (list, tuple)) or len(feats) != self.num_stages:
            raise RuntimeError(
                f"Encoder must return list/tuple with len={self.num_stages}. "
                f"Got type={type(feats)} len={len(feats) if isinstance(feats, (list, tuple)) else 'NA'}"
            )

        feats = list(feats)
        self.dipam_last_metrics = {}
        self.dipam_last_aux = {}

        for i in self.active_stage_indices:
            plugin = self.dipam_stages[str(i)]
            feats[i], m = plugin(feats[i])
            self.dipam_last_metrics[f"stage{i}"] = m
            self.dipam_last_aux[f"stage{i}"] = plugin.last_aux

        out = self.decoder(feats)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

class ContinuousPrototypeMemory(nn.Module):

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        beta: float = 0.25,
        usage_reg: bool = True,
        usage_tau: float = 1.0,
        usage_rho: float = 0.10,
        usage_ema_decay: float = 0.99,
        usage_eps: float = 1e-6,
        use_cosine_sim: bool = True,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.beta = float(beta)

        self.usage_reg = bool(usage_reg)
        self.usage_tau = float(usage_tau)
        self.usage_rho = float(usage_rho)
        self.usage_ema_decay = float(usage_ema_decay)
        self.usage_eps = float(usage_eps)
        self.use_cosine_sim = bool(use_cosine_sim)

        embed = torch.randn(self.num_codes, self.code_dim, dtype=torch.float32)
        embed = embed / (embed.norm(dim=1, keepdim=True) + 1e-8)
        self.embed = nn.Parameter(embed)

        self.register_buffer(
            "usage_ema",
            torch.full((self.num_codes,), 1.0 / self.num_codes, dtype=torch.float32),
        )

    def _compute_usage_diversity_loss(self, p: torch.Tensor):
        eps = self.usage_eps
        K = self.num_codes
        rho = float(min(max(self.usage_rho, 0.0), 1.0))

        p_ema = self.usage_ema.detach()
        u = torch.full_like(p_ema, 1.0 / K)
        p_mix = (1.0 - rho) * p_ema + rho * u
        p_tgt = 0.9 * p_mix + 0.1 * u
        p_tgt = p_tgt / (p_tgt.sum() + eps)

        div = (p * ((p + eps).log() - (p_tgt + eps).log())).sum()
        ent = -(p * (p + eps).log()).sum()
        return div, ent

    def forward(self, z: torch.Tensor):
        assert z.dim() == 2 and z.size(1) == self.code_dim

        z_fp32 = z.float()
        if self.use_cosine_sim:
            z_assign = z_fp32 / (z_fp32.norm(dim=1, keepdim=True) + 1e-8)
            e = self.embed.float()
            e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        else:
            z_assign = z_fp32
            e = self.embed.float()

        sim = z_assign @ e.t()
        tau = max(float(self.usage_tau), 1e-6)
        weights = torch.softmax(sim / tau, dim=1)

        q_fp32 = weights @ e
        indices = torch.argmax(weights, dim=1)

        commit_per_token = self.beta * ((q_fp32 - z_assign.detach()) ** 2).mean(dim=1)
        commit_loss = commit_per_token.mean()

        avg_probs = weights.mean(dim=0)
        avg_probs = _distributed_sum_(avg_probs)
        avg_probs = avg_probs / (avg_probs.sum() + 1e-10)

        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        if self.usage_reg:
            if self.training:
                with torch.no_grad():
                    self.usage_ema.mul_(self.usage_ema_decay).add_(
                        avg_probs.detach(),
                        alpha=1.0 - self.usage_ema_decay,
                    )
                    self.usage_ema.copy_(self.usage_ema / (self.usage_ema.sum() + self.usage_eps))
            div_loss, usage_entropy = self._compute_usage_diversity_loss(avg_probs)
        else:
            div_loss = torch.zeros((), device=z.device, dtype=torch.float32)
            usage_entropy = torch.zeros((), device=z.device, dtype=torch.float32)

        return (
            q_fp32.to(dtype=z.dtype),
            indices,
            commit_loss,
            perplexity,
            div_loss,
            usage_entropy,
            commit_per_token,
        )

class NonAxisMemory(nn.Module):

    def __init__(
        self,
        in_channels: int,
        stage_idx: int,
        num_stages: int,
        memory_type: Literal["discrete", "continuous"] = "discrete",
        embed_dim: int = 64,
        num_codes_global: int = 24,
        num_codes_local: int = 48,
        decay: float = 0.99,
        beta: float = 0.25,
        eps: float = 1e-5,
        usage_reg: bool = True,
        usage_lambda_global: float = 1e-3,
        usage_lambda_local: float = 1e-3,
        usage_tau: float = 1.0,
        usage_rho: float = 0.10,
        usage_ema_decay: float = 0.99,
        usage_topm: int = 32,
        use_layernorm: bool = True,
        pool_mode: Literal["mean", "attn", "topk_attn"] = "topk_attn",
        local_grid_3d: Tuple[int, int, int] = (2, 2, 2),
        attn_temperature: float = 1.0,
        topk_k: int = 8,
        topk_ratio: float | None = None,
        topk_min: int = 4,
        topk_max: int = 12,
        learnable_global_scale: bool = True,
        global_scale_max: float = 1.0,
        global_scale_init: float = 0.35,
        learnable_local_scale: bool = True,
        local_scale_max: float = 1.0,
        local_scale_init: float = 0.12,
        residual_gate: bool = True,
        residual_gate_mode: Literal["voxel", "channel"] = "channel",
        residual_gate_detach_R: bool = True,
        residual_gate_detach_F: bool = False,
        residual_gate_init_bias: float = 3.8,
        ctrl_dim: int = 8,
        t_local: Optional[int] = None,
        c_mid_global: Optional[int] = None,
        c_mid_local: Optional[int] = None,
    ):
        super().__init__()
        C = int(in_channels)
        d = int(embed_dim)

        self.in_channels = C
        self.stage_idx = int(stage_idx)
        self.num_stages = int(num_stages)
        self.memory_type = str(memory_type)

        self.pool_mode = pool_mode
        self.local_grid_3d = tuple(int(x) for x in local_grid_3d)
        self.attn_temperature = float(attn_temperature)
        self.topk_k = int(topk_k)
        self.topk_ratio = None if topk_ratio is None else float(topk_ratio)
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)

        self.usage_reg = bool(usage_reg)
        self.usage_lambda_global = float(usage_lambda_global)
        self.usage_lambda_local = float(usage_lambda_local)

        self.learnable_global_scale = bool(learnable_global_scale)
        self.global_scale_max = float(global_scale_max)
        self.learnable_local_scale = bool(learnable_local_scale)
        self.local_scale_max = float(local_scale_max)

        self.residual_gate = bool(residual_gate)
        self.residual_gate_mode = residual_gate_mode
        self.residual_gate_detach_R = bool(residual_gate_detach_R)
        self.residual_gate_detach_F = bool(residual_gate_detach_F)

        t_local = int(t_local) if t_local is not None else min(16, max(8, C // 16))
        c_mid_global = int(c_mid_global) if c_mid_global is not None else min(24, max(16, C // 12))
        c_mid_local = int(c_mid_local) if c_mid_local is not None else min(24, max(16, C // 12))

        self.t_local = t_local
        self.c_mid_global = c_mid_global
        self.c_mid_local = c_mid_local

        self.attn = nn.Conv3d(C, 1, kernel_size=1, bias=True)

        self.proj = nn.Linear(C, d, bias=True)
        self.norm = nn.LayerNorm(d) if use_layernorm else nn.Identity()

        if self.memory_type == "discrete":
            mem_cls = EMAVectorQuantizer
            mem_kwargs = dict(
                decay=decay,
                eps=eps,
                beta=beta,
                usage_reg=usage_reg,
                usage_tau=usage_tau,
                usage_rho=usage_rho,
                usage_ema_decay=usage_ema_decay,
                usage_topm=usage_topm,
            )
        elif self.memory_type == "continuous":
            mem_cls = ContinuousPrototypeMemory
            mem_kwargs = dict(
                beta=beta,
                usage_reg=usage_reg,
                usage_tau=usage_tau,
                usage_rho=usage_rho,
                usage_ema_decay=usage_ema_decay,
            )
        else:
            raise ValueError(f"Unknown memory_type={memory_type}")

        self.mem_global = mem_cls(num_codes_global, d, **mem_kwargs)
        self.mem_local = mem_cls(num_codes_local, d, **mem_kwargs)

        self.global_to_c = nn.Linear(d, C, bias=True)
        self.global_refine = nn.Sequential(
            nn.Conv3d(C, c_mid_global, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(
                c_mid_global,
                c_mid_global,
                kernel_size=3,
                padding=1,
                groups=c_mid_global,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv3d(c_mid_global, C, kernel_size=1, bias=True),
        )

        self.local_to_t = nn.Linear(d, t_local, bias=True)
        self.local_refine = nn.Sequential(
            nn.Conv3d(t_local, t_local, kernel_size=3, padding=1, groups=t_local, bias=False),
            nn.GELU(),
            nn.Conv3d(t_local, c_mid_local, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(c_mid_local, C, kernel_size=1, bias=True),
        )

        self.stage_embed = nn.Embedding(self.num_stages, int(ctrl_dim))
        self.ctrl_mlp = nn.Sequential(
            nn.Linear(C + int(ctrl_dim), 16, bias=True),
            nn.GELU(),
            nn.Linear(16, 3, bias=True),
        )

        g_init = max(1e-6, min(float(global_scale_init), self.global_scale_max))
        g_p = min(max(g_init / self.global_scale_max, 1e-6), 1.0 - 1e-6)
        g_raw = torch.tensor(float(torch.log(torch.tensor(g_p / (1.0 - g_p))).item()), dtype=torch.float32)
        self.global_scale_raw = nn.Parameter(g_raw) if self.learnable_global_scale else None
        if not self.learnable_global_scale:
            self.register_buffer("global_scale_raw_buffer", g_raw)

        l_init = max(1e-6, min(float(local_scale_init), self.local_scale_max))
        l_p = min(max(l_init / self.local_scale_max, 1e-6), 1.0 - 1e-6)
        l_raw = torch.tensor(float(torch.log(torch.tensor(l_p / (1.0 - l_p))).item()), dtype=torch.float32)
        self.local_scale_raw = nn.Parameter(l_raw) if self.learnable_local_scale else None
        if not self.learnable_local_scale:
            self.register_buffer("local_scale_raw_buffer", l_raw)

        if self.residual_gate:
            out_ch = 1 if residual_gate_mode == "voxel" else C
            self.res_gate = nn.Sequential(
                nn.Conv3d(C * 2, C, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv3d(C, out_ch, kernel_size=1, bias=True),
            )
            nn.init.constant_(self.res_gate[-1].bias, float(residual_gate_init_bias))

        self.last_aux: Dict[str, Any] = {}

    def _clear_auxiliary_memory_outputs(self):
        self.last_aux = {}

    def _compute_k(self, L: int) -> int:
        if self.topk_ratio is None:
            return max(1, min(self.topk_k, int(L)))
        k = int(round(self.topk_ratio * L))
        k = max(self.topk_min, min(k, self.topk_max))
        return max(1, min(k, int(L)))

    def _get_bounded_scale(self, raw_name: str, max_val: float, dtype, device):
        if raw_name == "global":
            raw = self.global_scale_raw if self.learnable_global_scale else self.global_scale_raw_buffer
        else:
            raw = self.local_scale_raw if self.learnable_local_scale else self.local_scale_raw_buffer
        val = float(max_val) * torch.sigmoid(raw.to(device=device))
        return val.to(dtype=dtype)

    def _extract_global_memory_token(self, F_in: torch.Tensor):
        B, C, D, H, W = F_in.shape

        if self.pool_mode == "mean":
            tok = F_in.mean(dim=(2, 3, 4)).view(B, 1, C)
            return tok

        logits = self.attn(F_in).view(B, D * H * W)
        feats = F_in.view(B, C, D * H * W)
        tau = max(self.attn_temperature, 1e-6)

        if self.pool_mode == "attn":
            w = torch.softmax(logits.float() / tau, dim=1).to(dtype=F_in.dtype)
            tok = (feats * w.unsqueeze(1)).sum(dim=2).view(B, 1, C)
            return tok

        if self.pool_mode == "topk_attn":
            L = D * H * W
            k_eff = self._compute_k(L)
            vals, idx = torch.topk(logits.float(), k_eff, dim=1, largest=True, sorted=False)
            w = torch.softmax(vals / tau, dim=1).to(dtype=F_in.dtype)
            feats_k = torch.gather(feats, dim=2, index=idx.unsqueeze(1).expand(B, C, k_eff))
            tok = (feats_k * w.unsqueeze(1)).sum(dim=2).view(B, 1, C)
            return tok

        raise ValueError(f"Unknown pool_mode={self.pool_mode}")

    def _extract_local_memory_tokens(self, F_in: torch.Tensor):
        pooled = F.adaptive_avg_pool3d(F_in, output_size=self.local_grid_3d)
        B, C, gd, gh, gw = pooled.shape
        tok = pooled.flatten(2).transpose(1, 2).contiguous()
        return tok, (gd, gh, gw)

    def _compute_stage_adaptive_mixing(self, F_in: torch.Tensor):
        B, C, _, _, _ = F_in.shape
        gap = F_in.mean(dim=(2, 3, 4))
        sid = torch.full((B,), self.stage_idx, dtype=torch.long, device=F_in.device)
        se = self.stage_embed(sid)
        ctrl = self.ctrl_mlp(torch.cat([gap, se], dim=1))

        base_g = self._get_bounded_scale("global", self.global_scale_max, F_in.dtype, F_in.device)
        base_l = self._get_bounded_scale("local", self.local_scale_max, F_in.dtype, F_in.device)

        alpha = (base_g * torch.sigmoid(ctrl[:, 0])).view(B, 1, 1, 1, 1)
        beta = (base_l * torch.sigmoid(ctrl[:, 1])).view(B, 1, 1, 1, 1)
        gate_bias = ctrl[:, 2].view(B, 1, 1, 1, 1)
        return alpha, beta, gate_bias

    def _compute_residual_memory_gate(
        self,
        F_in: torch.Tensor,
        R: torch.Tensor,
        gate_bias: Optional[torch.Tensor] = None,
    ):
        Fin = F_in.detach() if self.residual_gate_detach_F else F_in
        Rin = R.detach() if self.residual_gate_detach_R else R
        logits = self.res_gate(torch.cat([Fin, Rin], dim=1))
        if gate_bias is not None:
            logits = logits + gate_bias
        g = torch.sigmoid(logits)
        if self.residual_gate_mode == "voxel":
            g = g.expand_as(R)
        return g

    def forward(self, F_in: torch.Tensor):
        assert F_in.dim() == 5, f"Expected [B,C,D,H,W], got {tuple(F_in.shape)}"
        self._clear_auxiliary_memory_outputs()

        B, C, D, H, W = F_in.shape

        Tg = self._extract_global_memory_token(F_in)
        Ug = self.norm(self.proj(Tg))
        qg, idx_g, loss_g, ppl_g, div_g, ent_g, cpt_g = self.mem_global(Ug.reshape(-1, Ug.shape[-1]))
        qg = qg.view_as(Ug)

        Rg_vec = self.global_to_c(qg.squeeze(1)).view(B, C, 1, 1, 1)
        R_global = self.global_refine(Rg_vec.expand(B, C, D, H, W))

        Tl, grid = self._extract_local_memory_tokens(F_in)
        Ul = self.norm(self.proj(Tl))
        ql, idx_l, loss_l, ppl_l, div_l, ent_l, cpt_l = self.mem_local(Ul.reshape(-1, Ul.shape[-1]))
        ql = ql.view_as(Ul)

        gd, gh, gw = grid
        Rt = self.local_to_t(ql).view(B, gd, gh, gw, self.t_local)
        Rt = Rt.permute(0, 4, 1, 2, 3).contiguous()
        Rt = F.interpolate(Rt, size=(D, H, W), mode="trilinear", align_corners=False)
        R_local = self.local_refine(Rt)

        alpha, beta, gate_bias = self._compute_stage_adaptive_mixing(F_in)
        R = alpha * R_global + beta * R_local

        if self.residual_gate:
            R = R * self._compute_residual_memory_gate(F_in, R, gate_bias=gate_bias)

        F_out = F_in + R

        self.last_aux = {
            "shape": (B, C, D, H, W),
            "num_codes_global": int(self.mem_global.num_codes),
            "num_codes_local": int(self.mem_local.num_codes),
            "global": {
                "none": {
                    "indices": idx_g.view(B, 1),
                    "commit": cpt_g.view(B, 1),
                }
            },
            "local": {
                "none": {
                    "indices": idx_l.view(B, gd * gh * gw),
                    "commit": cpt_l.view(B, gd * gh * gw),
                    "grid": grid,
                }
            },
        }

        vq_commit_global = loss_g
        vq_commit_local = loss_l
        vq_div_global = div_g if self.usage_reg else F_in.new_zeros(())
        vq_div_local = div_l if self.usage_reg else F_in.new_zeros(())

        vq_loss = (
            vq_commit_global
            + vq_commit_local
            + self.usage_lambda_global * vq_div_global
            + self.usage_lambda_local * vq_div_local
        )

        rr_g = ((R_global.flatten(1).norm(dim=1)) / (F_in.flatten(1).norm(dim=1) + 1e-6)).mean().detach()
        rr_l = ((R_local.flatten(1).norm(dim=1)) / (F_in.flatten(1).norm(dim=1) + 1e-6)).mean().detach()
        rr_t = ((R.flatten(1).norm(dim=1)) / (F_in.flatten(1).norm(dim=1) + 1e-6)).mean().detach()

        metrics = DiPAMMetrics(
            vq_loss=vq_loss,
            vq_commit_global=vq_commit_global,
            vq_commit_local=vq_commit_local,
            vq_div_global=vq_div_global,
            vq_div_local=vq_div_local,
            perplexity_x_global=ppl_g.detach(),
            perplexity_y_global=ppl_g.detach(),
            perplexity_z_global=ppl_g.detach(),
            perplexity_x_local=ppl_l.detach(),
            perplexity_y_local=ppl_l.detach(),
            perplexity_z_local=ppl_l.detach(),
            residual_ratio_global=rr_g,
            residual_ratio_local=rr_l,
            residual_ratio_total=rr_t,
            alpha_mean=alpha.mean().detach(),
            beta_mean=beta.mean().detach(),
            usage_entropy_global=ent_g.detach(),
            usage_entropy_local=ent_l.detach(),
        )

        return F_out, metrics

class NonAxisMemoryNet(nn.Module):

    def __init__(
        self,
        base_net: nn.Module,
        enc_channels: Sequence[int],
        plugin_stage_indices: Optional[Sequence[int]] = None,
        plugin_stage_mask: Optional[Sequence[bool]] = None,
        memory_type: Literal["discrete", "continuous"] = "discrete",
        memory_embed_dim: int = 64,
        memory_num_codes_global: int = 24,
        memory_num_codes_local: int = 48,
        memory_decay: float = 0.99,
        memory_beta: float = 0.25,
        memory_usage_reg: bool = True,
        memory_usage_lambda_global: float = 1e-3,
        memory_usage_lambda_local: float = 1e-3,
        memory_usage_tau: float = 1.0,
        memory_usage_rho: float = 0.10,
        memory_usage_ema_decay: float = 0.99,
        memory_usage_topm: int = 32,
        memory_global_pool_mode: Literal["mean", "attn", "topk_attn"] = "topk_attn",
        memory_local_grid_3d: Tuple[int, int, int] = (2, 2, 2),
        memory_attn_temperature: float = 1.0,
        memory_topk_k: int = 8,
        memory_topk_ratio: float | None = None,
        memory_topk_min: int = 4,
        memory_topk_max: int = 12,
        memory_learnable_global_scale: bool = True,
        memory_global_scale_max: float = 1.0,
        memory_global_scale_init: float = 0.35,
        memory_learnable_local_scale: bool = True,
        memory_local_scale_max: float = 1.0,
        memory_local_scale_init: float = 0.12,
        memory_residual_gate: bool = True,
        memory_residual_gate_mode: Literal["voxel", "channel"] = "channel",
        memory_residual_gate_detach_R: bool = True,
        memory_residual_gate_detach_F: bool = False,
        memory_residual_gate_init_bias: float = 3.8,
        memory_ctrl_dim: int = 8,
        memory_t_local: Optional[int] = None,
        memory_c_mid_global: Optional[int] = None,
        memory_c_mid_local: Optional[int] = None,
    ):
        super().__init__()
        self.base = base_net
        self.enc_channels = list(map(int, enc_channels))

        enc_attr, dec_attr = DiPAMNet._resolve_enc_dec(self.base)
        self.encoder = getattr(self.base, enc_attr)
        self.decoder = getattr(self.base, dec_attr)

        self.num_stages = len(self.enc_channels)

        self.active_stage_indices = DiPAMNet._resolve_active_stage_indices(
            num_stages=self.num_stages,
            plugin_stage_indices=plugin_stage_indices,
            plugin_stage_mask=plugin_stage_mask,
        )

        self.nonaxis_memory_stages = nn.ModuleDict()
        for i in self.active_stage_indices:
            self.nonaxis_memory_stages[str(i)] = NonAxisMemory(
                in_channels=int(self.enc_channels[i]),
                stage_idx=i,
                num_stages=self.num_stages,
                memory_type=memory_type,
                embed_dim=memory_embed_dim,
                num_codes_global=memory_num_codes_global,
                num_codes_local=memory_num_codes_local,
                decay=memory_decay,
                beta=memory_beta,
                usage_reg=memory_usage_reg,
                usage_lambda_global=memory_usage_lambda_global,
                usage_lambda_local=memory_usage_lambda_local,
                usage_tau=memory_usage_tau,
                usage_rho=memory_usage_rho,
                usage_ema_decay=memory_usage_ema_decay,
                usage_topm=memory_usage_topm,
                pool_mode=memory_global_pool_mode,
                local_grid_3d=memory_local_grid_3d,
                attn_temperature=memory_attn_temperature,
                topk_k=memory_topk_k,
                topk_ratio=memory_topk_ratio,
                topk_min=memory_topk_min,
                topk_max=memory_topk_max,
                learnable_global_scale=memory_learnable_global_scale,
                global_scale_max=memory_global_scale_max,
                global_scale_init=memory_global_scale_init,
                learnable_local_scale=memory_learnable_local_scale,
                local_scale_max=memory_local_scale_max,
                local_scale_init=memory_local_scale_init,
                residual_gate=memory_residual_gate,
                residual_gate_mode=memory_residual_gate_mode,
                residual_gate_detach_R=memory_residual_gate_detach_R,
                residual_gate_detach_F=memory_residual_gate_detach_F,
                residual_gate_init_bias=memory_residual_gate_init_bias,
                ctrl_dim=memory_ctrl_dim,
                t_local=memory_t_local,
                c_mid_global=memory_c_mid_global,
                c_mid_local=memory_c_mid_local,
            )

        self.memory_last_metrics: Dict[str, DiPAMMetrics] = {}
        self.memory_last_aux: Dict[str, Dict[str, Any]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if not isinstance(feats, (list, tuple)) or len(feats) != self.num_stages:
            raise RuntimeError(
                f"Encoder must return list/tuple with len={self.num_stages}. "
                f"Got type={type(feats)} len={len(feats) if isinstance(feats, (list, tuple)) else 'NA'}"
            )

        feats = list(feats)
        self.memory_last_metrics = {}
        self.memory_last_aux = {}

        for i in self.active_stage_indices:
            plugin = self.nonaxis_memory_stages[str(i)]
            feats[i], m = plugin(feats[i])
            self.memory_last_metrics[f"stage{i}"] = m
            self.memory_last_aux[f"stage{i}"] = plugin.last_aux

        out = self.decoder(feats)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out
