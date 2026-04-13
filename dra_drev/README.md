# Open3DAD — 3D Point Cloud Anomaly Detection

A unified framework for **3D point cloud anomaly detection** supporting **Open-Industry** and **Anomaly-ShapeNet / Real3D-AD** datasets.

Two training paradigms are included:
- **DRA** (Dual-head Reference-Augmented) — 4-head architecture with reference set comparison, seen/pseudo/composite anomaly heads.
- **DevNet** (Deviation Network) — single-head MIL-style scoring with deviation loss.

Both use **Point-MAE** or **Point-BERT** as the 3D backbone (via PatchCore feature extraction).

## Project Structure

```
.
├── DRA_train.py              # DRA training entry
├── DevNet_train.py           # DevNet training entry
├── DRA_eval_p.py             # DRA evaluation (sample-level + point-level AUC)
├── DevNet_eval_p.py          # DevNet evaluation (sample-level + point-level AUC)
├── generate_table_all_metrics.py  # Aggregate eval logs into CSV tables
├── dataloaders/
│   ├── dataloader.py         # Unified DataLoader builder
│   ├── utlis.py              # BalancedBatchSampler (optional)
│   └── datasets/
│       ├── base_dataset.py   # Base class
│       ├── open_industry.py  # Open-Industry dataset
│       ├── anomaly_shapenet.py  # Anomaly-ShapeNet / Real3D-AD dataset
│       ├── transform.py      # Point cloud augmentation (SphereCropMask, etc.)
│       └── untils.py         # Filename parsing, train/test split helpers
└── model/
    ├── DRA.py                # DRA model (4 heads)
    ├── DevNet.py             # DevNet model (single head)
    ├── loss/
    │   ├── __init__.py       # build_criterion()
    │   ├── deviation_loss.py
    │   └── binary_focal_loss.py
    └── pointmae/
        ├── patchcore/        # PatchCore feature extraction wrapper
        ├── feature_extractors/  # FPFH, raw, pc features
        ├── M3DM/             # Point-MAE / Point-BERT models, FPS, KNN
        └── utils/            # Visualization, preprocessing, etc.
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12 (with CUDA support recommended)
- open3d
- scikit-learn
- tqdm
- timm
- pandas
- openpyxl (for Excel export in eval scripts)
- matplotlib
- tensorboard (optional, for DRA training visualization)
- pointnet2_ops (for FPS CUDA kernel, optional — CPU fallback available)

Install core dependencies:

```bash
pip install torch torchvision open3d scikit-learn tqdm timm pandas openpyxl matplotlib tensorboard
```

### Point-MAE / Point-BERT Checkpoints

Place pretrained backbone weights under paths expected by `model/pointmae/M3DM/models.py`. Default paths:

- Point-MAE: `model/pointmae/point_mae_checkpoint/pretrain.pth`
- Point-BERT: `model/pointmae/point_mae_checkpoint/point_bert.pth`

See `Model1.__init__()` in `model/pointmae/M3DM/models.py` for the exact logic.

## Supported Datasets

| Dataset | `--dataset` flag | Description |
|---------|-----------------|-------------|
| Open-Industry | `open_industry` | Per-class folders with `train/` and `test/` containing `.pcd` files. Anomaly classes: Bump, Deformation, Dent, Scar, Scratch. |
| Anomaly-ShapeNet / Real3D-AD | `anomaly_shapenet` | `train/*.pcd` for normal, `GT/*.txt` for anomaly annotations (with point-level labels). Anomaly classes: bulge, broken, concavity, crak, scratch. |

### Dataset Directory Layout

**Open-Industry:**
```
dataset_root/
└── classname/
    ├── train/          # Normal samples: classname_001.pcd, ...
    └── test/           # Normal + anomaly samples: classname_Bump_001.pcd, ...
```

**Anomaly-ShapeNet:**
```
dataset_root/
└── classname/
    ├── train/          # Normal samples: *.pcd
    ├── test/
    │   └── good/       # Test normal samples: *.pcd
    └── GT/             # Anomaly annotations: *.txt (x,y,z,label per line)
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset type: `open_industry`, `anomaly_shapenet` | `open_industry` |
| `--dataset_root` | Path to dataset root directory | *required* |
| `--classname` | Object class to train on | — |
| `--know_class` | Known anomaly types (space-separated) | `None` |
| `--nAnomaly` | Number of anomaly samples per class in training set | `5` |
| `--xyz_backbone` | 3D backbone: `Point_MAE` or `Point_BERT` | `Point_MAE` |
| `--use_pseudo_anomaly` | Enable pseudo anomaly generation (1=yes, 0=no) | `1` (DRA) / `0` (DevNet) |
| `--ramdn_seed` | Random seed for reproducibility | `42` |
| `--nRef` | Number of reference samples (DRA only) | `5` |
| `--total_heads` | Number of scoring heads (DRA only) | `4` |
| `--topk` | Top-k ratio for MIL scoring | `0.1` |
| `--eval_ckpt` | Path to checkpoint for evaluation | — |
| `--experiment_dir` | Output directory for logs, checkpoints, results | `./experiment/` |
| `--device` | Device to use | `cuda:0` |

## Evaluation Output

Both evaluation scripts (`DRA_eval_p.py`, `DevNet_eval_p.py`) output:
- **Sample-level AUC**: Overall, Seen-only, Unseen-only (ROC + PR)
- **Point-level AUC**: Seen Point, Unseen Point (ROC + PR) — computed via token-gradient nearest-neighbor interpolation
- Results are also saved to an Excel file (`eval_results_all.xlsx`)

Use `generate_table_all_metrics.py` to aggregate multiple evaluation results into summary CSV tables.

## Feature Caching

The first run extracts Point-MAE/Point-BERT features for all samples and caches them as `.npz` files.
Subsequent runs detect existing caches and skip backbone loading, significantly reducing VRAM usage.

- **Open-Industry**: caches stored at `OpenIndustry_<Backbone>_feature/` (parallel to dataset directory)
- **Anomaly-ShapeNet**: caches stored alongside the dataset (configurable in code)

## Notes

- Training automatically saves the **Top-5** best checkpoints ranked by evaluation metric, removing the worst when exceeded.
- TensorBoard logging is enabled for DRA training (logs under `<experiment_dir>/tensorboard/`).
- The pseudo anomaly pipeline uses `SphereCropMask` augmentation to generate local geometric deformations (bumps/dents).

## License

This project is released for academic research purposes.
