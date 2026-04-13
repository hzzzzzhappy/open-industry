import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
import open3d as o3d
import datetime
import pandas as pd
import copy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataloaders.dataloader import initDataloader
from model.DevNet import DevNet
from model.loss import build_criterion
from model.pointmae.patchcore.patchcore import PatchCore
class Eval:
    """Standalone evaluator for running inference with a pretrained checkpoint."""

    def __init__(self, args):
        self.args = args
        kwargs = {'num_workers': args.workers}
        print("Preparing evaluation dataloader...")
        _, self.test_loader = initDataloader.build(args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        
        # Initialize DevNet
        self.model = DevNet(args).to(self.device)
        
        # Load weights
        if args.eval_ckpt:
            self.load_weights(args.eval_ckpt)
        
        self._init_feature_extractor()

    def load_weights(self, weight_path: str):
        if weight_path is None:
            raise ValueError("weight_path is required for evaluation")
        state = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        print(f"Load pretrain weight from: {weight_path}")

    def _init_feature_extractor(self):
        self.feature_extractor = None
        self._use_cached_features = self._check_all_cache_exists()
        if self._use_cached_features:
            print("[Eval] All feature caches found, skipping backbone loading.")
            return

        print("[Eval] Incomplete cache, loading feature extractor...")
        self.feature_extractor = PatchCore(self.device)

        basic_template_path = os.listdir(os.path.join(self.args.dataset_root, self.args.classname, 'train'))[0]
        basic_template_path = os.path.join(self.args.dataset_root, self.args.classname, 'train', basic_template_path)

        pcd_o3d = o3d.io.read_point_cloud(basic_template_path)
        points = np.asarray(pcd_o3d.points).astype(np.float32)
        center = np.mean(points, axis=0, keepdims=True)
        points_centered = points - center
        scale = np.max(np.linalg.norm(points_centered, axis=1))
        basic_template = (points_centered / (scale + 1e-6)).astype(np.float32)

        self.feature_extractor.load(
            backbone=None,
            layers_to_extract_from=[],
            device=self.device,
            input_shape=(1, 3, 224, 224),
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            basic_template=basic_template,
            xyz_backbone_name=self.args.xyz_backbone
        )
        self.basic_template = basic_template
        self.feature_extractor.set_deep_feature_extractor()

    def _check_all_cache_exists(self):
        """Check if all feature caches exist for every sample in the dataloader."""
        from pathlib import Path as _Path
        root = os.path.join(self.args.dataset_root, self.args.classname)
        xyz_backbone = getattr(self.args, 'xyz_backbone', 'Point_MAE')

        for loader in [self.test_loader]:
            ds = loader.dataset
            for index in range(len(ds)):
                parts = os.path.join(root, ds.pcds[index]).split('/')
                parts.pop(-2)
                pcd_path = '/'.join(parts)
                feature_path = pcd_path.replace("Open-Industry", f"OpenIndustry_{xyz_backbone}_feature")
                feature_path = os.path.splitext(feature_path)[0] + ".npz"
                if not os.path.exists(feature_path):
                    fp = _Path(feature_path)
                    parts_p = list(fp.parts)
                    parts_p = ["pseudo" if p == "train" else p for p in parts_p]
                    if not os.path.exists(str(_Path(*parts_p))):
                        return False
        return True

    @staticmethod
    def normalization(data):
        data = np.asarray(data, dtype=np.float32)
        dmin = data.min()
        dmax = data.max()
        if dmax - dmin < 1e-12:
            return np.zeros_like(data, dtype=np.float32)
        return (data - dmin) / (dmax - dmin)

    def _resolve_excel_path(self) -> tuple[str, str]:
        """Resolve shared Excel path and run name derived from pretrain_dir."""
        if self.args.eval_ckpt:
            ckpt_path = os.path.abspath(self.args.eval_ckpt)
            seed_dir = os.path.dirname(ckpt_path)
            class_dir = os.path.dirname(seed_dir)
            exp_root = os.path.dirname(class_dir)
            
            run_name = f"{os.path.basename(class_dir)}_{os.path.basename(seed_dir)}"
            return os.path.join(exp_root, "eval_results_all.xlsx"), run_name

        base = self.args.pretrain_dir or self.args.experiment_dir or "."
        if os.path.splitext(base)[1]:
            weight_dir = os.path.dirname(base) or "."
        else:
            weight_dir = base
        excel_root = os.path.dirname(weight_dir) or weight_dir or "."
        os.makedirs(excel_root, exist_ok=True)
        run_name = os.path.basename(weight_dir)
        final_dir = os.path.dirname(excel_root)
        return os.path.join(final_dir, "eval_results_all.xlsx"), run_name

    def _save_metrics_to_excel(self, metrics: dict):
        """Append one row of metrics to a shared Excel file named eval_results_all.xlsx."""
        excel_path, run_name = self._resolve_excel_path()
        excel_path = os.path.abspath(excel_path)

        metrics = dict(metrics)
        metrics.setdefault("run_name", run_name)
        try:
            if os.path.exists(excel_path):
                df_existing = pd.read_excel(excel_path)
                df_new = pd.concat([df_existing, pd.DataFrame([metrics])], ignore_index=True)
            else:
                df_new = pd.DataFrame([metrics])
            df_new.to_excel(excel_path, index=False)
            print(f"[Eval] Metrics saved to Excel: {excel_path}")
        except Exception as exc:
            print(f"[Eval] Warning: failed to save metrics to Excel ({excel_path}): {exc}")

    def run(self, save_point_scores_dir: str = None):
        """Evaluate sample-level AUC (overall/seen/unseen) and approximate point-level AUC via token-gradient nearest-neighbour interpolation."""
        import numpy as np

        self.model.eval()
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            self.feature_extractor.requires_grad_(False)
        self.model.requires_grad_(False)

        tbar = tqdm(self.test_loader, desc='[Eval]')
        
        total_pred = []
        total_target = []

        all_point_scores_seen = []
        all_point_gts_seen = []
        all_point_scores_unseen = []
        all_point_gts_unseen = []

        if save_point_scores_dir:
            dir_path = save_point_scores_dir
            if os.path.splitext(save_point_scores_dir)[1]:
                dir_path = os.path.dirname(save_point_scores_dir) or "."
            os.makedirs(dir_path, exist_ok=True)
            save_point_scores_dir = dir_path

        for it, sample in enumerate(tbar):
            points_label = sample['points_label'].squeeze(0).numpy()  # (N,)
            points = sample['coord'].squeeze(0).numpy()               # (N,3)
            target = torch.from_numpy(sample['label'].squeeze(0).numpy()).long().to(self.device)
            label = sample['label'].squeeze(0).numpy()

            if 'pcd_features' in sample and sample['pcd_features'] is not None \
                    and 'center_idx' in sample and sample['center_idx'] is not None:
                pcd_features = sample['pcd_features'].squeeze(0).numpy()
                center_idx = sample['center_idx'].squeeze(0)
                if isinstance(center_idx, torch.Tensor):
                    center_idx = center_idx.numpy()
            else:
                pcd = torch.from_numpy(points).float().to(self.device)
                assert pcd is not None and pcd.numel() > 0, "[ERROR] pcd is None or empty."
                with torch.no_grad():
                    pcd_features, center_idx = self.feature_extractor._embed_pointmae(pcd, detach=True)

            # center_idx -> numpy int64, shape (M,)
            if isinstance(center_idx, torch.Tensor):
                center_idx_np = center_idx.detach().cpu().numpy()
            else:
                center_idx_np = np.asarray(center_idx)
            center_idx_np = np.asarray(center_idx_np).reshape(-1).astype(np.int64)

            if isinstance(pcd_features, np.ndarray):
                feat = torch.from_numpy(pcd_features).float().to(self.device)
            else:
                feat = pcd_features.detach().float().to(self.device)

            if feat.dim() == 2:
                feat = feat.unsqueeze(0)   # [1,M,C]
            elif feat.dim() != 3:
                raise RuntimeError(f"[ERROR] Unexpected pcd_features shape: {feat.shape}")

            feat = feat.detach().clone().requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            if self.feature_extractor is not None:
                self.feature_extractor.zero_grad(set_to_none=True)

            # DevNet forward returns a list [score]
            outputs = self.model(feat) 
            score_tensor = outputs[0] # [1, 1] or [1]

            total_pred.append(score_tensor.detach().cpu().item())
            total_target.append(target.detach().cpu().item())

            # DevNet score is already a scalar (or close to it)
            score = score_tensor.mean()

            grad_feat = torch.autograd.grad(
                outputs=score,
                inputs=feat,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]  # [1,M,C]

            token_score = grad_feat.detach().norm(dim=-1).squeeze(0).cpu().numpy()  # (M,)

            from scipy.spatial import cKDTree

            N = points.shape[0]

            if center_idx_np.min() < 0 or center_idx_np.max() >= N:
                raise RuntimeError(
                    f"[ERROR] center_idx out of range: min={center_idx_np.min()}, max={center_idx_np.max()}, N={N}"
                )

            centers = points[center_idx_np]  # (M,3)
            M = centers.shape[0]
            if token_score.shape[0] != M:
                raise RuntimeError(f"[ERROR] token_score(M)={token_score.shape[0]} != centers(M)={M}")

            tree = cKDTree(centers)
            _, nn_idx = tree.query(points, k=1)  # nn_idx: (N,)
            point_score = token_score[nn_idx]  # (N,)

            ps_min = point_score.min()
            point_score = (point_score - ps_min) / (point_score.max() - ps_min + 1e-12)
            point_score = point_score.astype(np.float32)

            gt = (points_label != 0).astype(int)  # (N,)
            if label == 0 or label == 1:
                all_point_scores_seen.append(point_score.reshape(-1))
                all_point_gts_seen.append(gt.reshape(-1))
            if label == 0 or label == 2:
                all_point_scores_unseen.append(point_score.reshape(-1))
                all_point_gts_unseen.append(gt.reshape(-1))

        total_pred = np.array(total_pred)
        total_target = np.array(total_target)
        
        # Normalize predictions
        total_pred = self.normalization(total_pred)

        total_target = total_target.astype(int)
        overall_binary = (total_target != 0).astype(int)
        total_roc, total_pr = aucPerformance(total_pred, overall_binary)

        seen_mask = (total_target == 0) | (total_target == 1)
        seen_targets = total_target[seen_mask]
        seen_pred = total_pred[seen_mask]
        seen_binary = (seen_targets == 1).astype(int)
        if np.any(seen_binary == 1) and np.any(seen_binary == 0):
            seen_roc, seen_pr = aucPerformance(seen_pred, seen_binary, prt=False)
        else:
            seen_roc, seen_pr = float('nan'), float('nan')

        unseen_mask = (total_target == 0) | (total_target == 2)
        unseen_targets = total_target[unseen_mask]
        unseen_pred = total_pred[unseen_mask]
        unseen_binary = (unseen_targets == 2).astype(int)
        if np.any(unseen_binary == 1) and np.any(unseen_binary == 0):
            unseen_roc, unseen_pr = aucPerformance(unseen_pred, unseen_binary, prt=False)
        else:
            unseen_roc, unseen_pr = float('nan'), float('nan')

        if len(all_point_scores_seen) > 0:
            seen_ps = np.concatenate(all_point_scores_seen, axis=0)
            seen_pg = np.concatenate(all_point_gts_seen, axis=0)
            if seen_pg.max() == seen_pg.min():
                seen_point_roc, seen_point_pr = float('nan'), float('nan')
            else:
                seen_point_roc, seen_point_pr = aucPerformance(seen_ps, seen_pg, prt=False)
        else:
            seen_point_roc, seen_point_pr = float('nan'), float('nan')

        if len(all_point_scores_unseen) > 0:
            unseen_ps = np.concatenate(all_point_scores_unseen, axis=0)
            unseen_pg = np.concatenate(all_point_gts_unseen, axis=0)
            if unseen_pg.max() == unseen_pg.min():
                unseen_point_roc, unseen_point_pr = float('nan'), float('nan')
            else:
                unseen_point_roc, unseen_point_pr = aucPerformance(unseen_ps, unseen_pg, prt=False)
        else:
            unseen_point_roc, unseen_point_pr = float('nan'), float('nan')

        print(f"[Eval] Overall   ROC: {total_roc:.4f}, PR: {total_pr:.4f}")
        print(f"[Eval] Seen-only ROC: {seen_roc:.4f}, PR: {seen_pr:.4f}")
        print(f"[Eval] Unseen-only ROC: {unseen_roc:.4f}, PR: {unseen_pr:.4f}")
        print(f"[Eval-Point] Seen Point ROC: {seen_point_roc:.4f}, PR: {seen_point_pr:.4f}")
        print(f"[Eval-Point] Unseen Point ROC: {unseen_point_roc:.4f}, PR: {unseen_point_pr:.4f}")

        metrics_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": getattr(self.args, "dataset", ""),
            "classname": getattr(self.args, "classname", ""),
            "overall_roc": float(total_roc),
            "overall_pr": float(total_pr),
            "seen_roc": float(seen_roc),
            "seen_pr": float(seen_pr),
            "unseen_roc": float(unseen_roc),
            "unseen_pr": float(unseen_pr),
            "seen_point_roc": float(seen_point_roc),
            "seen_point_pr": float(seen_point_pr),
            "unseen_point_roc": float(unseen_point_roc),
            "unseen_point_pr": float(unseen_point_pr),
        }
        self._save_metrics_to_excel(metrics_row)

        return total_roc, total_pr, seen_roc, seen_pr, unseen_roc, unseen_pr, seen_point_roc, seen_point_pr, unseen_point_roc, unseen_point_pr

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print(f"AUC-ROC: {roc_auc:.4f}, AUC-PR: {ap:.4f}")
    return roc_auc, ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='open_industry', help="dataset name: open_industry, anomaly_shapenet")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment/experiment_14', help="dataset root")
    parser.add_argument('--classname', type=str, default='fangxiedianpian', help="dataset class")
    parser.add_argument("--nAnomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument(
        '--know_class',   
        nargs='+',        
        type=str,
        default=None
    )
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=4, help="number of head in training")
    parser.add_argument("--nRef", type=int, default=5, help="number of reference set")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")
    parser.add_argument('--xyz_backbone', type=str, default='Point_MAE', help="Backbone: Point_MAE or Point_BERT")
    parser.add_argument('--use_pseudo_anomaly', type=int, default=0, help="whether to use pseudo anomaly data (1=True, 0=False)")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument('--eval_only', action='store_true', help="run evaluation only with a pretrained checkpoint")
    parser.add_argument('--eval_ckpt', type=str, default=None, help="path to checkpoint used for eval_only mode")
    parser.add_argument('--result_path', type=str, default=None, help="file path to save evaluation scores; defaults to <experiment_dir>/result_eval.txt")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    SEED = int(args.ramdn_seed)
    print(f"[Seed] Using seed: {SEED}")
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"Device: {args.device}")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("batch_size:", args.batch_size)
    os.makedirs(args.experiment_dir, exist_ok=True)

    evaluator = Eval(args)
    evaluator.run()

