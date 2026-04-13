import os,sys,pdb
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from dataloaders.dataloader import initDataloader
from model.DRA import DRA
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from model.loss import build_criterion
import random
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import open3d as o3d
import subprocess
import os
import time
from model.pointmae.patchcore.patchcore import PatchCore
from torch.utils.tensorboard import SummaryWriter
import datetime
import math
import copy
WEIGHT_DIR = './weights'

class Eval:
    """Standalone evaluator for running inference with a pretrained checkpoint."""

    def __init__(self, args):
        self.args = args
        kwargs = {'num_workers': args.workers}
        print("Preparing evaluation dataloader...")
        _, self.test_loader = initDataloader.build(args, **kwargs)

        if self.args.total_heads == 4:
            print("Preparing reference dataloader for evaluation...")
            temp_args = copy.deepcopy(args)
            temp_args.batch_size = self.args.nRef
            temp_args.nAnomaly = 0
            self.ref_loader, _ = initDataloader.build(temp_args, **kwargs)
            assert len(self.ref_loader) % args.nRef == 0, \
                f"[ERROR] The size of ref_loader ({len(self.ref_loader)}) is not divisible by args.nRef ({args.nRef})."
            self.ref = iter(self.ref_loader)

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.model = DRA(args).to(self.device)
        self.criterion = build_criterion(args.criterion).to(self.device)
        self._init_feature_extractor()


    def _init_feature_extractor(self):
        self.feature_extractor = PatchCore(self.device)
        basic_template_path = os.listdir(os.path.join(self.args.dataset_root, self.args.classname, 'train'))[0]
        basic_template_path = os.path.join(self.args.dataset_root, self.args.classname, 'train',basic_template_path)

        pcd_o3d = o3d.io.read_point_cloud(basic_template_path)

        points = np.asarray(pcd_o3d.points).astype(np.float32)

        center = np.mean(points, axis=0, keepdims=True)  # (1,3)
        points_centered = points - center                # (N,3)

        scale = np.max(np.linalg.norm(points_centered, axis=1))
        points_norm = points_centered / (scale + 1e-6)

        basic_template = points_norm.astype(np.float32)

        self.feature_extractor.load(
            backbone=None,
            layers_to_extract_from=[],
            device=self.device,
            input_shape=(1, 3, 224, 224),
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            basic_template = basic_template,
            xyz_backbone_name = self.args.xyz_backbone
        )
        self.basic_template = basic_template

        self.feature_extractor.set_deep_feature_extractor()

    @staticmethod
    def generate_target(target, eval=False):
        targets = list()
        if eval:
            targets.append(target == 0)
            targets.append(target)
            targets.append(target)
            targets.append(target)
            return targets
        temp_t = target != 0
        targets.append(target == 0)
        targets.append(temp_t[target != 2])
        targets.append(temp_t[target != 1])
        targets.append(target != 0)
        return targets

    @staticmethod
    def normalization(data):
        data = np.asarray(data, dtype=np.float32)
        dmin = data.min()
        dmax = data.max()
        if dmax - dmin < 1e-12:
            return np.zeros_like(data, dtype=np.float32)
        return (data - dmin) / (dmax - dmin)

    def load_weights(self, weight_path: str):
        if weight_path is None:
            raise ValueError("weight_path is required for evaluation")
        # Prefer safe loading; fall back for older PyTorch that lacks weights_only
        state = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        print(f"Load pretrain weight from: {weight_path}")

    def _resolve_excel_path(self) -> tuple[str, str]:
        """Resolve shared Excel path and run name derived from pretrain_dir."""
        base = self.args.pretrain_dir or self.args.experiment_dir or "."
        # Drop filename if it looks like one, then use its parent so multiple seeds merge
        if os.path.splitext(base)[1]:
            weight_dir = os.path.dirname(base) or "."
        else:
            weight_dir = base
        excel_root = os.path.dirname(weight_dir) or weight_dir or "."
        os.makedirs(excel_root, exist_ok=True)
        run_name = os.path.basename(weight_dir)
        return os.path.join(excel_root, "eval_results_all.xlsx"), run_name

    def _save_metrics_to_excel(self, metrics: dict):
        """Append one row of metrics to a shared Excel file named eval_results_all.xlsx."""
        excel_path, run_name = self._resolve_excel_path()
        excel_path = os.path.join(os.path.dirname(excel_path), "..", "eval_results_all.xlsx")
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
        """
        Evaluate sample-level AUC (overall/seen/unseen) and approximate
        point-level AUC via token-gradient nearest-neighbour interpolation.

        Args:
            save_point_scores_dir: optional path to save per-sample point_score npy files
        """
        import os
        import numpy as np
        import torch
        from tqdm import tqdm
        self.model.eval()
        if hasattr(self, "feature_extractor") and hasattr(self.feature_extractor, "eval"):
            self.feature_extractor.eval()

        if hasattr(self, "feature_extractor"):
            self.feature_extractor.requires_grad_(False)
        self.model.requires_grad_(False)

        tbar = tqdm(self.test_loader, desc='[Eval]')
        class_pred = [np.array([]) for _ in range(self.args.total_heads)]
        total_target = np.array([])

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
            pcd = torch.from_numpy(points).float().to(self.device)
            assert pcd is not None and pcd.numel() > 0, "[ERROR] pcd is None or empty."

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
            if hasattr(self, "feature_extractor") and hasattr(self.feature_extractor, "zero_grad"):
                self.feature_extractor.zero_grad(set_to_none=True)

            if self.args.total_heads == 4:
                try:
                    ref_pcd_feature = next(self.ref)['pcd_features']
                except StopIteration:
                    self.ref = iter(self.ref_loader)
                    ref_pcd_feature = next(self.ref)['pcd_features']

                if isinstance(ref_pcd_feature, np.ndarray):
                    ref = torch.from_numpy(ref_pcd_feature).float().to(self.device)
                else:
                    ref = ref_pcd_feature.detach().float().to(self.device)

                # ref -> [nRef, M, C]
                if ref.dim() == 4:
                    ref = ref.squeeze(0)
                if ref.dim() == 2:
                    ref = ref.unsqueeze(0)
                if ref.dim() != 3:
                    raise RuntimeError(f"[ERROR] Unexpected ref feature shape: {ref.shape}")

                assert ref.size(0) == self.args.nRef, \
                    f"[ERROR] ref_pcd_feature batch mismatch: got {ref.size(0)}, expected {self.args.nRef}"

                pcd_in = torch.cat([ref, feat], dim=0)
            else:
                pcd_in = feat  # [1, M, C]

            outputs = self.model(pcd_in, target, training=False)  # list of heads

            for head_idx in range(self.args.total_heads):
                out = outputs[head_idx]
                if head_idx == 0:
                    data = -1 * out.detach().cpu().numpy()
                else:
                    data = out.detach().cpu().numpy()
                class_pred[head_idx] = np.append(class_pred[head_idx], data)
            total_target = np.append(total_target, target.detach().cpu().numpy())

            score = 0.0
            for h in range(self.args.total_heads):
                s = outputs[h]
                if h == 0:
                    s = -s
                score = score + s.mean()

            grad_feat = torch.autograd.grad(
                outputs=score,
                inputs=feat,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]  # [1,M,C]

            token_score_t = grad_feat.detach().norm(dim=-1).squeeze(0)  # (M,) on GPU

            points_t = torch.from_numpy(points.astype(np.float32)).to(self.device)  # (N,3)
            N = points_t.shape[0]

            if center_idx_np.min() < 0 or center_idx_np.max() >= N:
                raise RuntimeError(
                    f"[ERROR] center_idx out of range: min={center_idx_np.min()}, max={center_idx_np.max()}, N={N}"
                )

            center_idx_t = torch.from_numpy(center_idx_np).long().to(self.device)
            centers_t = points_t[center_idx_t]  # (M,3)

            M = centers_t.shape[0]
            if token_score_t.shape[0] != M:
                raise RuntimeError(f"[ERROR] token_score(M)={token_score_t.shape[0]} != centers(M)={M}")

            dists = torch.cdist(points_t.unsqueeze(0), centers_t.unsqueeze(0)).squeeze(0)  # (N, M)
            nn = dists.argmin(dim=1)  # (N,)
            point_score_t = token_score_t[nn]  # (N,)

            ps_min = point_score_t.min()
            point_score_t = (point_score_t - ps_min) / (point_score_t.max() - ps_min + 1e-12)
            point_score = point_score_t.cpu().numpy().astype(np.float32)

            gt = (points_label != 0).astype(int)  # (N,)
            if label == 0 or label == 1:
                all_point_scores_seen.append(point_score.reshape(-1))
                all_point_gts_seen.append(gt.reshape(-1))
            if label == 0 or label == 2:
                all_point_scores_unseen.append(point_score.reshape(-1))
                all_point_gts_unseen.append(gt.reshape(-1))

            del pcd, pcd_features, center_idx, feat, pcd_in, outputs, grad_feat, token_score_t, point_score_t, points_t, dists
            torch.cuda.empty_cache()

        total_pred = self.normalization(class_pred[0])
        for head_idx in range(1, self.args.total_heads):
            total_pred = total_pred + self.normalization(class_pred[head_idx])

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
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")


    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
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
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--xyz_backbone', type=str, default='Point_MAE', help="Backbone: Point_MAE or Point_BERT")

    parser.add_argument('--use_pseudo_anomaly', type=int, default=1, help="whether to use pseudo anomaly data (1=True, 0=False)")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument('--eval_only', action='store_true', help="run evaluation only with a pretrained checkpoint")
    parser.add_argument('--eval_ckpt', type=str, default=None, help="path to checkpoint used for eval_only mode")
    parser.add_argument('--result_path', type=str, default=None, help="file path to save evaluation scores; defaults to <experiment_dir>/result_eval.txt")
    args = parser.parse_args()

    # ----------------- Reproducibility (controlled by --ramdn_seed) -----------------
    SEED = int(args.ramdn_seed)
    print(f"[Seed] Using seed: {SEED}")
    # Make hash-based operations deterministic
    os.environ['PYTHONHASHSEED'] = str(SEED)
    # For certain CUDA/cuBLAS nondeterministic ops (optional, may affect perf)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Python, numpy, torch RNGs
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Enforce deterministic algorithms where possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Device: {args.device}")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("batch_size:", args.batch_size)
    os.makedirs(args.experiment_dir, exist_ok=True)

    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    if args.eval_only:
        evaluator = Eval(args)
        weight_path = args.eval_ckpt or args.pretrain_dir
        if weight_path is None:
            raise ValueError("Eval mode requires --eval_ckpt or --pretrain_dir.")
        evaluator.load_weights(weight_path)
        result_path = args.result_path or os.path.join(args.experiment_dir, "result_eval.txt")
        evaluator.run(save_point_scores_dir=result_path)
        sys.exit(0)
