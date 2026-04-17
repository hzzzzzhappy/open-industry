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
````

---

## Requirements

* Python >= 3.8
* PyTorch >= 1.12
* CUDA is recommended
* open3d
* scikit-learn
* tqdm
* timm
* pandas
* openpyxl
* matplotlib
* tensorboard (optional, for DRA logging)
* pointnet2_ops (optional, for CUDA FPS acceleration; CPU fallback is available)

Install the main dependencies with:

```bash
pip install torch torchvision open3d scikit-learn tqdm timm pandas openpyxl matplotlib tensorboard
```

---

## Pretrained Backbones

The framework supports **Point-MAE** and **Point-BERT** as the 3D backbone.

Please place pretrained checkpoints under the paths expected by `model/pointmae/M3DM/models.py`.

Default paths:

* Point-MAE: `model/pointmae/point_mae_checkpoint/pretrain.pth`
* Point-BERT: `model/pointmae/point_mae_checkpoint/point_bert.pth`

Please refer to `Model1.__init__()` in `model/pointmae/M3DM/models.py` for the exact loading logic.

---

## Datasets

| Dataset                      | `--dataset` flag   | Description                                               |
| ---------------------------- | ------------------ | --------------------------------------------------------- |
| Open-Industry                | `open_industry`    | Industrial 3D anomaly detection dataset used in our paper |
| Anomaly-ShapeNet / Real3D-AD | `anomaly_shapenet` | Public benchmark supported by the unified dataloader      |

### OpenIndustry

OpenIndustry is an industrial 3D anomaly detection dataset introduced in our work.

At the current stage, we release the dataset for benchmarking and experimentation. In the open-set setting, a subset of anomaly types is treated as known during training, while the remaining anomaly types are reserved for unseen evaluation.

Typical anomaly categories include:

* Bump
* Deformation
* Dent
* Scar
* Scratch

Dataset download:
[https://huggingface.co/datasets/HanzheL/open-industry](https://huggingface.co/datasets/HanzheL/open-industry)

Directory layout:

```text
dataset_root/
└── classname/
    ├── train/          # normal samples: classname_001.pcd, ...
    └── test/           # normal + anomaly samples: classname_Bump_001.pcd, ...
```

### Anomaly-ShapeNet / Real3D-AD

This dataset is also supported under the unified dataloader.

Typical anomaly categories include:

* bulge
* broken
* concavity
* crak
* scratch

Directory layout:

```text
dataset_root/
└── classname/
    ├── train/          # normal samples: *.pcd
    ├── test/
    │   └── good/       # normal test samples: *.pcd
    └── GT/             # anomaly annotations: *.txt (x, y, z, label)
```

---

## Key Arguments

| Argument               | Description                                             | Default                      |
| ---------------------- | ------------------------------------------------------- | ---------------------------- |
| `--dataset`            | Dataset type: `open_industry`, `anomaly_shapenet`       | `open_industry`              |
| `--dataset_root`       | Path to dataset root directory                          | required                     |
| `--classname`          | Object class to train on                                | —                            |
| `--know_class`         | Known anomaly types (space-separated)                   | `None`                       |
| `--nAnomaly`           | Number of anomaly samples per class in the training set | `5`                          |
| `--xyz_backbone`       | Backbone: `Point_MAE` or `Point_BERT`                   | `Point_MAE`                  |
| `--use_pseudo_anomaly` | Enable pseudo anomaly generation                        | `1` for DRA / `0` for DevNet |
| `--ramdn_seed`         | Random seed                                             | `42`                         |
| `--nRef`               | Number of reference samples (DRA only)                  | `5`                          |
| `--total_heads`        | Number of scoring heads (DRA only)                      | `4`                          |
| `--topk`               | Top-k ratio for MIL scoring                             | `0.1`                        |
| `--eval_ckpt`          | Checkpoint path for evaluation                          | —                            |
| `--experiment_dir`     | Output directory                                        | `./experiment/`              |
| `--device`             | Device identifier                                       | `cuda:0`                     |

---

## Training

### Train DRA

```bash
python DRA_train.py \
  --dataset open_industry \
  --dataset_root /path/to/dataset \
  --classname bagel \
  --know_class Bump Dent \
  --nAnomaly 5 \
  --xyz_backbone Point_MAE \
  --device cuda:0
```

### Train DevNet

```bash
python DevNet_train.py \
  --dataset open_industry \
  --dataset_root /path/to/dataset \
  --classname bagel \
  --know_class Bump Dent \
  --nAnomaly 5 \
  --xyz_backbone Point_MAE \
  --device cuda:0
```

---

## Evaluation

### Evaluate DRA

```bash
python DRA_eval_p.py \
  --dataset open_industry \
  --dataset_root /path/to/dataset \
  --classname bagel \
  --eval_ckpt /path/to/checkpoint.pth \
  --device cuda:0
```

### Evaluate DevNet

```bash
python DevNet_eval_p.py \
  --dataset open_industry \
  --dataset_root /path/to/dataset \
  --classname bagel \
  --eval_ckpt /path/to/checkpoint.pth \
  --device cuda:0
```

---

## Evaluation Metrics and Outputs

Both evaluation scripts output:

* **Sample-level AUC**

  * Overall ROC-AUC / PR-AUC
  * Seen-only ROC-AUC / PR-AUC
  * Unseen-only ROC-AUC / PR-AUC

* **Point-level AUC**

  * Seen Point ROC-AUC / PR-AUC
  * Unseen Point ROC-AUC / PR-AUC

Point-level localization is computed through token-gradient nearest-neighbor interpolation.

Evaluation results are saved to:

* console logs
* `eval_results_all.xlsx`

To aggregate results across experiments, use:

```bash
python generate_table_all_metrics.py
```

---

## Feature Caching

During the first run, the framework extracts Point-MAE / Point-BERT features and caches them as `.npz` files.

On subsequent runs, cached features are automatically reused, which:

* avoids repeated backbone forwarding
* reduces GPU memory usage
* speeds up training and evaluation

Cache locations:

* **Open-Industry**: stored under `OpenIndustry_<Backbone>_feature/` parallel to the dataset directory
* **Anomaly-ShapeNet**: stored alongside the dataset (configurable in code)

---

## Additional Notes

* Training keeps the **Top-5** checkpoints ranked by evaluation metric and removes worse ones automatically.
* **TensorBoard** logging is enabled for DRA under `<experiment_dir>/tensorboard/`.
* The pseudo anomaly pipeline uses **SphereCropMask** to generate local geometric perturbations such as bump- or dent-like structures.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{liang2026opensetsupervised3danomaly,
      title={Open-Set Supervised 3D Anomaly Detection: An Industrial Dataset and a Generalisable Framework for Unknown Defects},
      author={Hanzhe Liang and Luocheng Zhang and Junyang Xia and HanLiang Zhou and Bingyang Guo and Yingxi Xie and Can Gao and Ruiyun Yu and Jinbao Wang and Pan Li},
      year={2026},
      eprint={2604.01171},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.01171},
}
```

---

## License

This project is released for academic research purposes.

```
