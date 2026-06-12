# Di-PAMNet

Official implementation of **Di-PAMNet: Lesion-Aware Axis-Decomposed Quantized Memory for Robust 3D Pancreatic Tumor Segmentation under Clinical Distribution Shift**.

Di-PAMNet is a 3D CT segmentation framework that integrates axis-decomposed dual-scale quantized memory with lesion-aware optimization for robust pancreatic tumor segmentation.

## Repository Structure
 ```
Di-PAMNet/
├── nnunetv2/
│   ├── nets/
│   │   └── dipamnet.py
│   └── training/
│       └── nnUNetTrainer/
│           └── variants/
│               └── dipamnet/
│                   └── nnUNetTrainerDiPAMNet.py
├── README.md
└── .gitignore
 ```

## Requirements
The code was developed for a PyTorch-based nnU-Net v2 environment.
Main dependencies:

python >= 3.9
torch
numpy
scipy
scikit-image
SimpleITK
batchgenerators
nnunetv2

## Acknowledgements 
We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.

