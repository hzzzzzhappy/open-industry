# Open-Set Supervised 3D Anomaly Detection

This repository accompanies our preprint: **Open-Set Supervised 3D Anomaly Detection: An Industrial Dataset and a Generalisable Framework for Unknown Defects**  [arXiv:2604.01171](https://arxiv.org/abs/2604.01171)

We study **open-set supervised 3D anomaly detection** for industrial point clouds, with a focus on detecting **unknown defects** that are not observed during training.

At the current stage, this repository provides **open-source baseline implementations** and the **OpenIndustry** dataset. Our full proposed method will be released later.

![OpenIndustry Overview](docs/dataset.png)

---

## Current Release

This repository currently includes:

- **OpenIndustry** dataset
- **Baseline implementations**
  - **DRA** (Dual-head Reference-Augmented)
  - **DevNet** (Deviation Network)

The implementation supports:
- **Open-Industry**
- **Anomaly-ShapeNet / Real3D-AD**
- **Point-MAE** and **Point-BERT** backbones
- **sample-level** and **point-level** evaluation
- **seen / unseen** anomaly split evaluation

### Status

- [x] OpenIndustry dataset
- [x] Baseline implementations
- [ ] Our full method

---

## Overview

Open-set supervised 3D anomaly detection considers the setting where training data contains only a subset of anomaly types, while testing includes both **seen** and **unseen** defects.

This repository provides benchmark baselines used in our paper for this setting, together with the OpenIndustry dataset.

---

## Included Baselines

### DRA
**DRA (Dual-head Reference-Augmented)** is a 4-head architecture with reference-set comparison and multiple anomaly scoring heads, including seen/pseudo/composite anomaly learning.

It supports pseudo anomaly synthesis through local geometric perturbation.

### DevNet
**DevNet (Deviation Network)** is a single-head anomaly detection baseline trained with deviation loss under an MIL-style scoring scheme.

### Our Method
Our full proposed method is **not included in the current release** and will be open-sourced in a later update.

---

## Project Structure

```text
.
├── DRA_train.py                     # DRA training entry
├── DevNet_train.py                  # DevNet training entry
├── DRA_eval_p.py                    # DRA evaluation (sample-level + point-level AUC)
├── DevNet_eval_p.py                 # DevNet evaluation (sample-level + point-level AUC)
├── generate_table_all_metrics.py    # Aggregate evaluation logs into CSV tables
├── dataloaders/
│   ├── dataloader.py                # Unified DataLoader builder
│   ├── utlis.py                     # BalancedBatchSampler (optional)
│   └── datasets/
│       ├── base_dataset.py          # Base dataset class
│       ├── open_industry.py         # Open-Industry dataset
│       ├── anomaly_shapenet.py      # Anomaly-ShapeNet / Real3D-AD dataset
│       ├── transform.py             # Point cloud augmentation (e.g., SphereCropMask)
│       └── untils.py                # Filename parsing and split helpers
└── model/
    ├── DRA.py                       # DRA model
    ├── DevNet.py                    # DevNet model
    ├── loss/
    │   ├── __init__.py
    │   ├── deviation_loss.py
    │   └── binary_focal_loss.py
    └── pointmae/
        ├── patchcore/               # PatchCore-style feature extraction wrapper
        ├── feature_extractors/      # FPFH / raw / point cloud features
        ├── M3DM/                    # Point-MAE / Point-BERT models, FPS, KNN
        └── utils/                   # Visualization and preprocessing utilities
