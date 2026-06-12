from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from nnunetv2.nets.dipamnet import DiPAMNet, EMAVectorQuantizer


class nnUNetTrainerDiPAMNet(nnUNetTrainer):
    DIPAM_STAGE_INDICES = [-1]

    DIPAM_EMBED_DIM = 64
    DIPAM_NUM_CODES_GLOBAL = 8
    DIPAM_NUM_CODES_LOCAL = 16
    DIPAM_DECAY = 0.99
    DIPAM_BETA = 0.06

    DIPAM_LEARNABLE_GLOBAL_SCALE = True
    DIPAM_GLOBAL_SCALE_MAX = 1.0
    DIPAM_GLOBAL_SCALE_INIT = 0.35

    DIPAM_LEARNABLE_LOCAL_SCALE = True
    DIPAM_LOCAL_SCALE_MAX = 1.0
    DIPAM_LOCAL_SCALE_INIT = 0.12

    DIPAM_GLOBAL_POOL_MODE = "topk_attn"
    DIPAM_LOCAL_POOL_MODE = "mean"
    DIPAM_ATTN_TEMPERATURE = 1.0
    DIPAM_TOPK_K = 8
    DIPAM_TOPK_RATIO = None
    DIPAM_TOPK_MIN = 4
    DIPAM_TOPK_MAX = 8

    DIPAM_LOCAL_GRID_X = (2, 2)
    DIPAM_LOCAL_GRID_Y = (2, 2)
    DIPAM_LOCAL_GRID_Z = (2, 2)

    DIPAM_USE_AXIS_SOFTMAX_GATING = True
    DIPAM_EDGE_AWARE_GATE = True
    DIPAM_EDGE_K = 0.20
    DIPAM_GATE_DETACH_EDGE = True

    DIPAM_RESIDUAL_GATE = True
    DIPAM_RESIDUAL_GATE_MODE = "channel"
    DIPAM_RESIDUAL_GATE_DETACH_R = True
    DIPAM_RESIDUAL_GATE_DETACH_F = False
    DIPAM_RESIDUAL_GATE_INIT_BIAS = 3.8

    DIPAM_USAGE_REG = True
    DIPAM_USAGE_LAMBDA_GLOBAL = 1e-2
    DIPAM_USAGE_LAMBDA_LOCAL = 2e-3
    DIPAM_USAGE_TAU = 1.0
    DIPAM_USAGE_RHO = 0.20
    DIPAM_USAGE_EMA_DECAY = 0.99
    DIPAM_USAGE_TOPM = 8

    DIPAM_CTRL_DIM = 8
    DIPAM_TRI_DIM = None
    DIPAM_T_LOCAL = None
    DIPAM_C_MID_GLOBAL = None
    DIPAM_C_MID_LOCAL = None

    TUMOR_LABEL_IDS = (1,)
    TUMOR_CLASS_INDEX = 1
    LESION_MEMORY_LAMBDA = 0.07
    CODE_SEPARATION_LAMBDA = 0.004
    RECALL_AUXILIARY_LAMBDA = 0.40
    TUMOR_COMMIT_WEIGHT = 3.0
    BACKGROUND_COMMIT_WEIGHT = 1.0
    USE_RECALL_AUXILIARY_LOSS = True

    CODE_SEPARATION_TAU = 0.5
    CODE_SEPARATION_START_EPOCH = 50
    CODE_SEPARATION_RAMP_EPOCHS = 100
    CODE_SEPARATION_DECAY_START_EPOCH = 180
    CODE_SEPARATION_DECAY_EPOCHS = 80
    CODE_SEPARATION_FINAL_FACTOR = 0.25

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [
            s if ((s / 2**5) >= 1 and (s / 2**5) % 1 == 0)
            else round(s / 2**5 + 0.5) * 2**5
            for s in original_patch_size
        ]
        self.configuration_manager.configuration["patch_size"] = new_patch_size
        self.plans_manager.plans["configurations"][self.configuration_name]["patch_size"] = new_patch_size
        self.print_to_log_file(f"Patch size: {original_patch_size} -> {new_patch_size}")
        self.grad_scaler = None
        self.initial_lr = 5e-3
        self.weight_decay = getattr(self, "weight_decay", 5e-5)
        self.num_epochs = 300
        self.lambda_vq_max = 0.012
        self.lambda_vq_warmup_epochs = 0
        self.lambda_vq_ramp_epochs = 25

    def _get_network_for_attr(self):
        return self.network.module if hasattr(self.network, "module") else self.network

    def _set_dipam_ema_enabled(self, enabled: bool):
        for m in self.network.modules():
            if isinstance(m, EMAVectorQuantizer):
                m.ema_enabled = bool(enabled)

    def get_lambda_vq(self) -> float:
        e = int(self.current_epoch)
        if e < self.lambda_vq_warmup_epochs:
            return 0.0
        t = float(
            np.clip(
                (e - self.lambda_vq_warmup_epochs) / max(1, self.lambda_vq_ramp_epochs),
                0.0,
                1.0,
            )
        )
        return float(self.lambda_vq_max * t)

    def get_lambda_code_separation(self) -> float:
        e = int(self.current_epoch)
        start = int(self.CODE_SEPARATION_START_EPOCH)
        ramp = int(self.CODE_SEPARATION_RAMP_EPOCHS)
        decay_start = int(self.CODE_SEPARATION_DECAY_START_EPOCH)
        decay_epochs = int(self.CODE_SEPARATION_DECAY_EPOCHS)
        max_lam = float(self.CODE_SEPARATION_LAMBDA)
        final_lam = max_lam * float(self.CODE_SEPARATION_FINAL_FACTOR)
        if e < start:
            return 0.0
        if e < start + ramp:
            t = (e - start) / max(1, ramp)
            return float(max_lam * np.clip(t, 0.0, 1.0))
        if e < decay_start:
            return max_lam
        if e < decay_start + decay_epochs:
            t = (e - decay_start) / max(1, decay_epochs)
            t = float(np.clip(t, 0.0, 1.0))
            return float((1.0 - t) * max_lam + t * final_lam)
        return final_lam

    @staticmethod
    def _infer_enc_channels_from_base(
        base: nn.Module,
        configuration_manager: ConfigurationManager,
    ):
        for enc_name in ("encoder", "conv_encoder"):
            enc = getattr(base, enc_name, None)
            if enc is None:
                continue
            for attr in ("output_channels", "stage_channels", "features_per_stage"):
                val = getattr(enc, attr, None)
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    try:
                        return [int(x) for x in val]
                    except Exception:
                        pass
            stages = getattr(enc, "stages", None)
            if isinstance(stages, (list, tuple, nn.ModuleList)) and len(stages) >= 2:
                chans = []
                for st in stages:
                    last_conv = None
                    for m in st.modules():
                        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                            last_conv = m
                    if last_conv is not None:
                        chans.append(int(last_conv.out_channels))
                if len(chans) >= 2:
                    return chans
        cfg = configuration_manager.configuration
        base_features = cfg.get(
            "UNet_base_num_features",
            cfg.get("unet_base_num_features", cfg.get("base_num_features", 32)),
        )
        max_features = cfg.get("unet_max_num_features", cfg.get("UNet_max_num_features", 320))
        n_stages = len(cfg.get("conv_kernel_sizes", []))
        if n_stages < 2:
            n_stages = len(cfg.get("pool_op_kernel_sizes", [])) + 1
        if n_stages < 2:
            n_stages = 6
        return [min(int(base_features) * (2**i), int(max_features)) for i in range(n_stages)]

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        base = nnUNetTrainerNoDeepSupervision.build_network_architecture(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_input_channels=num_input_channels,
            enable_deep_supervision=enable_deep_supervision,
        )
        T = nnUNetTrainerDiPAMNet
        enc_channels = T._infer_enc_channels_from_base(base, configuration_manager)
        return DiPAMNet(
            base_net=base,
            enc_channels=enc_channels,
            plugin_stage_indices=T.DIPAM_STAGE_INDICES,
            dipam_embed_dim=T.DIPAM_EMBED_DIM,
            dipam_num_codes_global=T.DIPAM_NUM_CODES_GLOBAL,
            dipam_num_codes_local=T.DIPAM_NUM_CODES_LOCAL,
            dipam_code_sep_tau=T.CODE_SEPARATION_TAU,
            dipam_decay=T.DIPAM_DECAY,
            dipam_beta=T.DIPAM_BETA,
            dipam_usage_reg=T.DIPAM_USAGE_REG,
            dipam_usage_lambda_global=T.DIPAM_USAGE_LAMBDA_GLOBAL,
            dipam_usage_lambda_local=T.DIPAM_USAGE_LAMBDA_LOCAL,
            dipam_usage_tau=T.DIPAM_USAGE_TAU,
            dipam_usage_rho=T.DIPAM_USAGE_RHO,
            dipam_usage_ema_decay=T.DIPAM_USAGE_EMA_DECAY,
            dipam_usage_topm=T.DIPAM_USAGE_TOPM,
            dipam_global_pool_mode=T.DIPAM_GLOBAL_POOL_MODE,
            dipam_local_pool_mode=T.DIPAM_LOCAL_POOL_MODE,
            dipam_attn_temperature=T.DIPAM_ATTN_TEMPERATURE,
            dipam_topk_k=T.DIPAM_TOPK_K,
            dipam_topk_ratio=T.DIPAM_TOPK_RATIO,
            dipam_topk_min=T.DIPAM_TOPK_MIN,
            dipam_topk_max=T.DIPAM_TOPK_MAX,
            dipam_local_grid_x=T.DIPAM_LOCAL_GRID_X,
            dipam_local_grid_y=T.DIPAM_LOCAL_GRID_Y,
            dipam_local_grid_z=T.DIPAM_LOCAL_GRID_Z,
            dipam_learnable_global_scale=T.DIPAM_LEARNABLE_GLOBAL_SCALE,
            dipam_global_scale_max=T.DIPAM_GLOBAL_SCALE_MAX,
            dipam_global_scale_init=T.DIPAM_GLOBAL_SCALE_INIT,
            dipam_learnable_local_scale=T.DIPAM_LEARNABLE_LOCAL_SCALE,
            dipam_local_scale_max=T.DIPAM_LOCAL_SCALE_MAX,
            dipam_local_scale_init=T.DIPAM_LOCAL_SCALE_INIT,
            dipam_use_axis_softmax_gating=T.DIPAM_USE_AXIS_SOFTMAX_GATING,
            dipam_edge_aware_gate=T.DIPAM_EDGE_AWARE_GATE,
            dipam_edge_k=T.DIPAM_EDGE_K,
            dipam_gate_detach_edge=T.DIPAM_GATE_DETACH_EDGE,
            dipam_residual_gate=T.DIPAM_RESIDUAL_GATE,
            dipam_residual_gate_mode=T.DIPAM_RESIDUAL_GATE_MODE,
            dipam_residual_gate_detach_R=T.DIPAM_RESIDUAL_GATE_DETACH_R,
            dipam_residual_gate_detach_F=T.DIPAM_RESIDUAL_GATE_DETACH_F,
            dipam_residual_gate_init_bias=T.DIPAM_RESIDUAL_GATE_INIT_BIAS,
            dipam_ctrl_dim=T.DIPAM_CTRL_DIM,
            dipam_tri_dim=T.DIPAM_TRI_DIM,
            dipam_t_local=T.DIPAM_T_LOCAL,
            dipam_c_mid_global=T.DIPAM_C_MID_GLOBAL,
            dipam_c_mid_local=T.DIPAM_C_MID_LOCAL,
        )

    def _get_dipam_vq_loss(self):
        net = self._get_network_for_attr()
        metrics = getattr(net, "dipam_last_metrics", None)
        if not isinstance(metrics, dict) or len(metrics) == 0:
            return None
        vq_loss_total = None
        for m in metrics.values():
            if m is None:
                continue
            vq_loss_total = m.vq_loss if vq_loss_total is None else vq_loss_total + m.vq_loss
        return vq_loss_total

    def _loss_requires_lists(self) -> bool:
        mod = type(self.loss).__module__
        name = type(self.loss).__name__
        return ("deep_supervision" in mod) or ("DeepSupervision" in name)

    def _harmonize_output_target(self, output, target):
        ds = self._loss_requires_lists()
        out_is_list = isinstance(output, (list, tuple))
        tgt_is_list = isinstance(target, (list, tuple))
        if ds:
            output = list(output) if out_is_list else [output]
            if not tgt_is_list:
                target = [target] * len(output)
            else:
                target = list(target)
            if len(target) < len(output):
                target = target + [target[-1]] * (len(output) - len(target))
            elif len(target) > len(output):
                target = target[: len(output)]
            return output, target
        output = output[0] if out_is_list else output
        target = target[0] if tgt_is_list else target
        return output, target

    def _unwrap_output_for_eval(self, output):
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise RuntimeError("Network returned empty list/tuple output.")
            return output[0]
        return output

    def _get_tumor_mask(self, target):
        t = target[0] if isinstance(target, list) else target
        if self.label_manager.has_regions:
            raise RuntimeError("Region-based targets are not supported by this Di-PAMNet trainer.")
        if t.ndim != 5 or t.shape[1] != 1:
            raise RuntimeError(f"Expected target [B,1,D,H,W], got {tuple(t.shape)}")
        tumor = torch.zeros_like(t, dtype=torch.float32)
        for lab in self.TUMOR_LABEL_IDS:
            tumor = torch.maximum(tumor, (t == int(lab)).float())
        return tumor

    def _get_active_aux_stage_key(self, aux_dict: dict) -> str:
        def _stage_num(k: str) -> int:
            s = str(k)
            if s.startswith("stage"):
                return int(s.replace("stage", ""))
            return -1
        keys = sorted(aux_dict.keys(), key=_stage_num)
        return keys[-1]

    def _local_axis_occupancy(self, tumor_mask, aux_local):
        B, _, D, H, W = tumor_mask.shape
        gx0, gx1 = aux_local["x"]["grid"]
        gy0, gy1 = aux_local["y"]["grid"]
        gz0, gz1 = aux_local["z"]["grid"]
        mx = tumor_mask.permute(0, 4, 1, 2, 3).reshape(B * W, 1, D, H)
        mx = F.adaptive_max_pool2d(mx, (gx0, gx1)).view(B, W, gx0 * gx1)
        my = tumor_mask.permute(0, 3, 1, 2, 4).reshape(B * H, 1, D, W)
        my = F.adaptive_max_pool2d(my, (gy0, gy1)).view(B, H, gy0 * gy1)
        mz = tumor_mask.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
        mz = F.adaptive_max_pool2d(mz, (gz0, gz1)).view(B, D, gz0 * gz1)
        return mx, my, mz

    def _weighted_commitment_loss(self, commit, occ):
        w = self.BACKGROUND_COMMIT_WEIGHT + (
            self.TUMOR_COMMIT_WEIGHT - self.BACKGROUND_COMMIT_WEIGHT
        ) * occ.float()
        return (w * commit).sum() / w.sum().clamp_min(1.0)

    def _soft_code_separation_loss(self, soft_assign, occ):
        if soft_assign is None:
            raise RuntimeError("Di-PAM local soft assignments are required for code separation loss.")
        if soft_assign.ndim != 4:
            raise RuntimeError(f"Expected soft_assign [B,N,G,K], got {tuple(soft_assign.shape)}")
        if occ.ndim != 3:
            raise RuntimeError(f"Expected occupancy [B,N,G], got {tuple(occ.shape)}")
        if soft_assign.shape[:3] != occ.shape:
            raise RuntimeError(f"Shape mismatch: soft_assign={tuple(soft_assign.shape)}, occ={tuple(occ.shape)}")
        pi = soft_assign.reshape(-1, soft_assign.shape[-1])
        m = occ.reshape(-1).float()
        tumor_mass = (pi * m[:, None]).sum(dim=0)
        background_mass = (pi * (1.0 - m[:, None])).sum(dim=0)
        if tumor_mass.sum().detach().item() < 1e-6:
            return pi.new_zeros(())
        if background_mass.sum().detach().item() < 1e-6:
            return pi.new_zeros(())
        tumor_hist = tumor_mass / tumor_mass.sum().clamp_min(1e-6)
        background_hist = background_mass / background_mass.sum().clamp_min(1e-6)
        return (tumor_hist * background_hist.detach()).sum()

    def _compute_lesion_aware_memory_losses(self, target):
        net = self._get_network_for_attr()
        aux_dict = getattr(net, "dipam_last_aux", None)
        if not isinstance(aux_dict, dict) or len(aux_dict) == 0:
            z = torch.zeros((), device=self.device)
            return z, z
        stage_key = self._get_active_aux_stage_key(aux_dict)
        aux = aux_dict[stage_key]
        B, _, D, H, W = aux["shape"]
        tumor = self._get_tumor_mask(target)
        tumor_ds = F.interpolate(tumor, size=(D, H, W), mode="nearest")
        mx, my, mz = self._local_axis_occupancy(tumor_ds, aux["local"])
        cxl = aux["local"]["x"]["commit"]
        cyl = aux["local"]["y"]["commit"]
        czl = aux["local"]["z"]["commit"]
        lesion_memory = (
            self._weighted_commitment_loss(cxl, mx)
            + self._weighted_commitment_loss(cyl, my)
            + self._weighted_commitment_loss(czl, mz)
        ) / 3.0
        pxl = aux["local"]["x"].get("soft_assign", None)
        pyl = aux["local"]["y"].get("soft_assign", None)
        pzl = aux["local"]["z"].get("soft_assign", None)
        code_separation = (
            self._soft_code_separation_loss(pxl, mx)
            + self._soft_code_separation_loss(pyl, my)
            + self._soft_code_separation_loss(pzl, mz)
        ) / 3.0
        return lesion_memory, code_separation

    def _recall_oriented_tumor_loss(self, output, target, alpha=0.3, beta=0.7, smooth=1e-5):
        if self.label_manager.has_regions:
            raise RuntimeError("Region-based targets are not supported by this Di-PAMNet trainer.")
        tumor = self._get_tumor_mask(target)
        prob = torch.softmax(output, dim=1)[:, self.TUMOR_CLASS_INDEX : self.TUMOR_CLASS_INDEX + 1]
        tp = (prob * tumor).sum(dim=(0, 2, 3, 4))
        fp = (prob * (1.0 - tumor)).sum(dim=(0, 2, 3, 4))
        fn = ((1.0 - prob) * tumor).sum(dim=(0, 2, 3, 4))
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - tversky.mean()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        lambda_vq = float(self.get_lambda_vq())
        lambda_sep = float(self.get_lambda_code_separation())
        self._set_dipam_ema_enabled(lambda_vq > 0.0)
        self.optimizer.zero_grad(set_to_none=True)
        output_raw = self.network(data)
        output_loss, target_loss = self._harmonize_output_target(output_raw, target)
        segmentation_loss = self.loss(output_loss, target_loss)
        vq_loss = self._get_dipam_vq_loss()
        vq_loss = lambda_vq * vq_loss if vq_loss is not None else 0.0
        lesion_memory_raw, code_separation_raw = self._compute_lesion_aware_memory_losses(target)
        lesion_memory_loss = self.LESION_MEMORY_LAMBDA * lesion_memory_raw
        code_separation_loss = lambda_sep * code_separation_raw
        if self.USE_RECALL_AUXILIARY_LOSS:
            output_for_aux = self._unwrap_output_for_eval(output_raw)
            recall_auxiliary_raw = self._recall_oriented_tumor_loss(output_for_aux, target)
            recall_auxiliary_loss = self.RECALL_AUXILIARY_LAMBDA * recall_auxiliary_raw
        else:
            recall_auxiliary_loss = torch.zeros((), device=self.device)
        loss = segmentation_loss + vq_loss + lesion_memory_loss + code_separation_loss + recall_auxiliary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu().item())}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        lambda_vq = float(self.get_lambda_vq())
        lambda_sep = float(self.get_lambda_code_separation())
        self._set_dipam_ema_enabled(False)
        try:
            output_raw = self.network(data)
            output_loss, target_loss = self._harmonize_output_target(output_raw, target)
            segmentation_loss = self.loss(output_loss, target_loss)
            vq_loss = self._get_dipam_vq_loss()
            vq_loss = lambda_vq * vq_loss if vq_loss is not None else 0.0
            lesion_memory_raw, code_separation_raw = self._compute_lesion_aware_memory_losses(target)
            lesion_memory_loss = self.LESION_MEMORY_LAMBDA * lesion_memory_raw
            code_separation_loss = lambda_sep * code_separation_raw
            if self.USE_RECALL_AUXILIARY_LOSS:
                output_for_aux = self._unwrap_output_for_eval(output_raw)
                recall_auxiliary_raw = self._recall_oriented_tumor_loss(output_for_aux, target)
                recall_auxiliary_loss = self.RECALL_AUXILIARY_LAMBDA * recall_auxiliary_raw
            else:
                recall_auxiliary_loss = torch.zeros((), device=self.device)
            loss = segmentation_loss + vq_loss + lesion_memory_loss + code_separation_loss + recall_auxiliary_loss
            output = self._unwrap_output_for_eval(output_raw)
            target_eval = target[0] if isinstance(target, list) else target
            axes = [0] + list(range(2, output.ndim))
            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg
            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target_eval != self.label_manager.ignore_label).float()
                    target_eval = target_eval.clone()
                    target_eval[target_eval == self.label_manager.ignore_label] = 0
                else:
                    mask = 1 - target_eval[:, -1:]
                    target_eval = target_eval[:, :-1]
            else:
                mask = None
            tp, fp, fn, _ = get_tp_fp_fn_tn(
                predicted_segmentation_onehot,
                target_eval,
                axes=axes,
                mask=mask,
            )
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
            return {
                "loss": float(loss.detach().cpu().item()),
                "tp_hard": tp_hard,
                "fp_hard": fp_hard,
                "fn_hard": fn_hard,
            }
        finally:
            self._set_dipam_ema_enabled(True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        for candidate in [
            getattr(self.network, "decoder", None),
            getattr(getattr(self.network, "base", None), "decoder", None),
            getattr(self._get_network_for_attr(), "decoder", None),
        ]:
            if candidate is not None and hasattr(candidate, "deep_supervision"):
                candidate.deep_supervision = False
