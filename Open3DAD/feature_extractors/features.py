"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

from sklearn import random_projection
from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds
import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from feature_extractors.pointnet2_utils import *
from pointnet2_ops import pointnet2_utils
import cv2
import os
from utils.mvtec3d_util import *
import time
import open3d as o3d
# from feature_extractors.models import *
from torch.utils.data import DataLoader
from knn_cuda import KNN

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    # breakpoint()
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

class Features(torch.nn.Module):


    def unorganized_data_to_organized(self,unorganized_pc, none_zero_data_list):
        '''

        Args:
            unorganized_pc:
            none_zero_data_list:

        Returns:

        '''
        # print(none_zero_data_list[0].shape)
        if not isinstance(none_zero_data_list, list):
            none_zero_data_list = [none_zero_data_list]

        for idx in range(len(none_zero_data_list)):
            none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().detach().cpu().numpy()

        # print("unorganized_pc",unorganized_pc.shape)


        unorganized_pc = unorganized_pc.numpy()
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
            

        full_data_list = []

        for none_zero_data in none_zero_data_list:
            if none_zero_data.ndim == 1:
                none_zero_data = np.expand_dims(none_zero_data,1)
            full_data = np.zeros((unorganized_pc.shape[0], none_zero_data.shape[1]), dtype=none_zero_data.dtype)
            
            if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
                full_data[nonzero_indices, :] = none_zero_data
            else:
                full_data = none_zero_data

            full_data_reshaped = full_data.reshape((1, unorganized_pc.shape[0], none_zero_data.shape[1]))
            full_data_tensor = torch.tensor(full_data_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
            full_data_list.append(full_data_tensor)

        return full_data_list

    def normalize(self,pred, max_value=None, min_value=None):
        if max_value is None or min_value is None:
            return (pred - pred.min()) / (pred.max() - pred.min())
        else:
            return (pred - min_value) / (max_value - min_value)

    def purify_abnormal_lib_by_normal_nn_delete(
        self,
        chunk: int = 2048,
        delete_ratio_cap: float = 1.0,
        verbose: bool = True,
    ):
        """
        For each patch in the normal library, find the closest 1 abnormal patch in the abnormal library,
        and delete these abnormal patches (after deduplication).

        Parameters:
            chunk: Compute cdist in chunks to prevent GPU OOM
            delete_ratio_cap: Maximum ratio of abnormal library to delete (0,1], 1.0 means no limit; 0.3 means delete at most 30%
            verbose: Print logs

        Side effects:
            - Update self.ab_patch_lib in-place
            - Save self.ab_delete_idx / self.ab_hits for debugging
        """
        assert hasattr(self, "patch_lib") and self.patch_lib is not None, "Please build normal library self.patch_lib first"
        assert hasattr(self, "ab_patch_lib") and self.ab_patch_lib is not None, "Please build abnormal library self.ab_patch_lib first"

        # Compatible with list / tensor
        if isinstance(self.patch_lib, list):
            self.patch_lib = torch.cat(self.patch_lib, 0)
        if isinstance(self.ab_patch_lib, list):
            self.ab_patch_lib = torch.cat(self.ab_patch_lib, 0)

        if self.patch_lib.numel() == 0 or self.ab_patch_lib.numel() == 0:
            if verbose:
                print("[Purify-NN-Delete] patch_lib or ab_patch_lib is empty, skip.")
            return

        if not (0 < delete_ratio_cap <= 1.0):
            raise ValueError(f"delete_ratio_cap must be in (0,1], current={delete_ratio_cap}")
        device = self.args.device if hasattr(self, "args") and hasattr(self.args, "device") else self.ab_patch_lib.device

        normal = self.patch_lib.to(device, non_blocking=True).float()      # (Nn, D)
        abnormal = self.ab_patch_lib.to(device, non_blocking=True).float() # (Na, D)

        Nn = normal.shape[0]
        Na = abnormal.shape[0]

        if verbose:
            print(f"[Purify-NN-Delete] Begin. normal={tuple(normal.shape)}, abnormal={tuple(abnormal.shape)}")

        # Count how many times each abnormal patch is "nearest neighbor hit" (optional but useful)
        hits = torch.zeros(Na, device=device, dtype=torch.int32)
        chosen_idx_list = []

        for s in range(0, Nn, chunk):
            e = min(s + chunk, Nn)
            dist = torch.cdist(normal[s:e], abnormal)   # (b, Na)
            nn_idx = dist.argmin(dim=1)                 # (b,)
            chosen_idx_list.append(nn_idx)
            hits.scatter_add_(0, nn_idx, torch.ones_like(nn_idx, dtype=torch.int32))

        chosen_idx = torch.cat(chosen_idx_list, dim=0)      # (Nn,)
        delete_idx_unique = torch.unique(chosen_idx)        # (Nd,)

        # Optional: limit maximum deletion ratio to avoid deleting too much at once
        max_delete = int(round(delete_ratio_cap * Na))
        if delete_idx_unique.numel() > max_delete:
            # If exceed limit, prioritize deleting those with most hits (most "normal-like" anomalies)
            del_scores = hits[delete_idx_unique].float()
            top = torch.topk(del_scores, k=max_delete, largest=True).indices
            delete_idx_unique = delete_idx_unique[top]

        # Build keep mask
        keep_mask = torch.ones(Na, device=device, dtype=torch.bool)
        keep_mask[delete_idx_unique] = False

        # Save debug info (put on CPU)
        self.ab_delete_idx = delete_idx_unique.detach().cpu()
        self.ab_hits = hits.detach().cpu()

        # Update abnormal library (keep remaining)
        ab_new = abnormal[keep_mask].contiguous()
        self.ab_patch_lib = ab_new  # Keep on device

        if verbose:
            print(f"[Purify-NN-Delete] Delete {delete_idx_unique.numel()} / {Na} abnormal patches.")
            print(f"[Purify-NN-Delete] New abnormal size = {self.ab_patch_lib.shape[0]}")
    def apply_ad_scoremap(self,image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)




    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.9,args = None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.deep_feature_extractor = Model(device=self.device)
        # self.deep_feature_extractor.to(self.device)
        # self.deep_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)
        self.args = args
        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []
        self.ab_patch_lib = []
        self.anomaly_patch_lib = []
        self.pre_patch_lib = []
        self.tmp_patch_lib = []
        self.name_list = []
        self.test_patch_lib = []


        self.normal_image_preds = []     # One normality image-level score per sample (float)
        self.normal_pixel_preds = []     # One flattened normality map per sample (np.ndarray, shape=[Ni])
        self.normal_predictions = []     # One original shape normality map per sample (np.ndarray)
        self.normal_paths = []           # (Optional) Path for each sample, convenient for alignment check



        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)

        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def init_para(self):
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0




    def compute_anomay_scores(self, patch, mask, label, path, unorganized_pc, unorganized_pc_no_zeros, center):
        # Note: check if self.patch_lib shape is 1000
        # Check how many patches there are
        # patch->(4096,128), self.patch_lib->(1000,128)
        dist = torch.cdist(patch, self.patch_lib)  
        min_val, min_idx = torch.min(dist, dim=1)

        feature_map_dims = patch.shape[0]
        s_map = min_val.view(1, 1, feature_map_dims)


        if self.args.use_LFSA:
            s_map = interpolating_points_chunked(unorganized_pc_no_zeros.permute(0,2,1).to(self.args.device), center.permute(0,2,1).to(self.args.device), s_map.to(self.args.device)).permute(0,2,1)
            s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0])

            if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
                s_map = s_map.squeeze().reshape(1,224,224)
                s_map = self.blur(s_map)

            else:
                num_group = 1024
                group_size = 12

                batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
                center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), num_group)  # B G 3

                # knn to get the neighborhood
                knn = KNN(k=group_size, transpose_mode=True)
                _, idx = knn(unorganized_pc_no_zeros, center)  # B G M

                ori_idx = idx
                idx_base = torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device).view(-1, 1, 1) * num_points
                
                idx = idx + idx_base
                idx = idx.view(-1)
                # breakpoint()
                idx = idx.to(s_map.device)
                neighborhood = s_map.reshape(batch_size * num_points, -1)[idx, :]
                neighborhood = neighborhood.reshape(batch_size, num_group, group_size, -1).contiguous()
                agg_s_map = torch.mean(neighborhood,-2).view(1, 1, -1)

                s_map = interpolating_points_chunked(unorganized_pc_no_zeros.permute(0,2,1).cuda(), center.permute(0,2,1).cuda(), agg_s_map.cuda()).permute(0,2,1)
                s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0])

        s_map = s_map.squeeze(0)
        s = torch.max(s_map)

        if self.args.dataset == 'real':
            s = torch.mean(s_map)
        if self.args.dataset == 'shapenet':
            tmp_s,_ = torch.topk(s_map, 80)
            s = torch.mean(tmp_s)
        if self.args.dataset == 'mulsen' or self.args.dataset == 'minishift' or self.args.dataset == 'quan':
            tmp_s,_ = torch.topk(s_map, 80)
            s = torch.mean(tmp_s)     
        # if self.args.dataset == 'mc3dad':
        #     tmp_s,_ = torch.topk(s_map, 80)
        #     s = torch.mean(tmp_s)

        if self.args.vis_save:
            while isinstance(path,list):
                path = path[0]
            from pathlib import Path
            parts = path.split("data", 1) 
            post_data_path = parts[1].lstrip(os.sep) 
            save_path = "./vis-results/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = normalize(s_map.squeeze())

            scoremap = (scoremap.cpu().numpy() * 255).astype(np.uint8)
            scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
            scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
            unorganized_pc = unorganized_pc.squeeze().cpu()
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)

            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())
            save_path = "./vis-results-GT/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = scoremap.cpu().numpy().astype(np.uint8)
            scoremap[mask.flatten().numpy()==1]=np.array([255,0,0])
            scoremap[mask.flatten().numpy()==0]=np.array([0,0,255])
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)
            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())

        
        self.image_preds.append(s.cpu().numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.cpu().flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())

        self.predictions.append(s_map.squeeze().detach().cpu().squeeze().numpy())
        self.gts.append(mask.squeeze().detach().cpu().squeeze().numpy())


    def compute_normality_scores(
        self,
        patch,
        mask,
        label,
        path,
        unorganized_pc,
        unorganized_pc_no_zeros,
        center,
    ):

        # patch -> (4096, 128), self.ab_patch_lib -> (1000, 128)
        # Note: this computes distance to "abnormal patch library", farther distance means more normal
        dist = torch.cdist(patch, self.ab_patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)

        feature_map_dims = patch.shape[0]
        s_map = min_val.view(1, 1, feature_map_dims)

        if self.args.use_LFSA:
            s_map = interpolating_points_chunked(
                unorganized_pc_no_zeros.permute(0, 2, 1).to(self.args.device),
                center.permute(0, 2, 1).to(self.args.device),
                s_map.to(self.args.device),
            ).permute(0, 2, 1)
            s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0])

            num_group = 1024
            group_size = 12

            batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
            center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), num_group)  # B G 3

            # knn to get the neighborhood
            knn = KNN(k=group_size, transpose_mode=True)
            _, idx = knn(unorganized_pc_no_zeros, center)  # B G M

            idx_base = (
                torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device)
                .view(-1, 1, 1)
                * num_points
            )

            idx = idx + idx_base
            idx = idx.view(-1).to(s_map.device)

            neighborhood = s_map.reshape(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.reshape(batch_size, num_group, group_size, -1).contiguous()
            agg_s_map = torch.mean(neighborhood, -2).view(1, 1, -1)

            s_map = interpolating_points_chunked(
                unorganized_pc_no_zeros.permute(0, 2, 1).cuda(),
                center.permute(0, 2, 1).cuda(),
                agg_s_map.cuda(),
            ).permute(0, 2, 1)
            s_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [s_map])[0])

        # s_map: (1, N) or (1, H, W), etc., uniformly squeeze to single sample map below
        s_map = s_map.squeeze(0)

        # ===== (Optional) If you want sample-wise normalization of normality before caching, uncomment this =====
        # eps = 1e-6
        # s_min = s_map.min()
        # s_max = s_map.max()
        # s_map = (s_map - s_min) / (s_max - s_min + eps)

        # image-level normality score (you can also change to mean/topk by dataset)
        s = torch.max(s_map)
        # ===== Key: only cache normality here, no longer do subtraction overwrite, no longer append to anomaly-related lists =====
        # 1) image-level normality
        self.normal_image_preds.append(float(s.detach().cpu().item()))

        # 2) map-level normality (original shape, aligned with predictions)
        norm_map_np = s_map.detach().cpu().squeeze().numpy()
        self.normal_predictions.append(norm_map_np)

        # 3) pixel/point-level normality (each sample flattened and stored separately to avoid alignment errors later)
        self.normal_pixel_preds.append(norm_map_np.reshape(-1))

        # 4) (Optional) Save path for checking if anomaly and normality correspond one-to-one
        while isinstance(path, list):
            path = path[0]
        self.normal_paths.append(path)
        
    def fuse_scores_and_store_back(self, lam=0.3, mode="minmax", eps=1e-6, clip_coef=(0.0, 1.0)):
        """
        Only fuse and overwrite:
            self.image_preds, self.pixel_preds

        Use normality (from abnormal library ab_patch_lib) to generate coefficient:
            coef = 1 - lam * s2

        Then modulate anomaly (from normal library patch_lib):
            final = anomaly * coef

        Normalization strategy (normalize separately, not pair-wise):
            - image_preds: normalize on the whole class (current batch of samples)
            - pixel_preds: normalize on the whole class (entire long vector)
            - normal_image_preds: normalize on the whole class
            - normal_pixel_preds: normalize on the whole class (concatenate then compute global min/max or mean/std)
        """

        import numpy as np

        num_samples = len(self.image_preds)
        assert len(self.normal_image_preds) == num_samples, \
            f"normal_image_preds({len(self.normal_image_preds)}) != image_preds({num_samples})"
        assert len(self.normal_pixel_preds) == num_samples, \
            f"normal_pixel_preds({len(self.normal_pixel_preds)}) != image_preds({num_samples})"

        # ============== helpers ==============
        def norm_minmax(x, mn, mx):
            return (x - mn) / (mx - mn + eps)

        def norm_zscore(x, mu, std):
            return (x - mu) / (std + eps)

        # ============== 1) image-level: normalize separately (class-wise) ==============
        img_anom = np.asarray(self.image_preds, dtype=np.float32).reshape(-1)          # anomaly
        img_norm = np.asarray(self.normal_image_preds, dtype=np.float32).reshape(-1)   # normality

        if mode == "minmax":
            a_mn, a_mx = float(img_anom.min()), float(img_anom.max())
            n_mn, n_mx = float(img_norm.min()), float(img_norm.max())
            img_anom_n = norm_minmax(img_anom, a_mn, a_mx)
            img_norm_n = norm_minmax(img_norm, n_mn, n_mx)
        elif mode == "zscore":
            a_mu, a_std = float(img_anom.mean()), float(img_anom.std())
            n_mu, n_std = float(img_norm.mean()), float(img_norm.std())
            img_anom_n = norm_zscore(img_anom, a_mu, a_std)
            img_norm_n = norm_zscore(img_norm, n_mu, n_std)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        coef_img = 1.0 - lam * img_norm_n
        if clip_coef is not None:
            lo, hi = clip_coef
            coef_img = np.clip(coef_img, lo, hi)

        img_final = img_anom_n * coef_img
        self.image_preds[:] = list(img_final.astype(np.float32))

        # ============== 2) pixel-level: normalize separately (class-wise) ==============
        # anomaly: entire long vector
        pix_anom = np.asarray(self.pixel_preds, dtype=np.float32).reshape(-1)

        # normality: concatenate each sample's flattened into one long vector (for computing global normalization parameters)
        pix_norm_all = np.concatenate(
            [np.asarray(v, dtype=np.float32).reshape(-1) for v in self.normal_pixel_preds],
            axis=0
        )

        if mode == "minmax":
            pa_mn, pa_mx = float(pix_anom.min()), float(pix_anom.max())
            pn_mn, pn_mx = float(pix_norm_all.min()), float(pix_norm_all.max())
            pix_anom_n_all = norm_minmax(pix_anom, pa_mn, pa_mx)
            pix_norm_n_all = norm_minmax(pix_norm_all, pn_mn, pn_mx)
        else:  # zscore
            pa_mu, pa_std = float(pix_anom.mean()), float(pix_anom.std())
            pn_mu, pn_std = float(pix_norm_all.mean()), float(pix_norm_all.std())
            pix_anom_n_all = norm_zscore(pix_anom, pa_mu, pa_std)
            pix_norm_n_all = norm_zscore(pix_norm_all, pn_mu, pn_std)

        # Note: pix_norm_n_all is concatenated, need to slice back by sample length, generate coef then multiply to corresponding anomaly segment
        pix_ptr = 0
        norm_ptr = 0

        # First write anomaly's normalized long vector to a mutable array
        pix_final_all = np.empty_like(pix_anom_n_all, dtype=np.float32)

        for i in range(num_samples):
            norm_flat = np.asarray(self.normal_pixel_preds[i], dtype=np.float32).reshape(-1)
            L = norm_flat.shape[0]

            # Anomaly segment (normalized)
            anom_seg = pix_anom_n_all[pix_ptr:pix_ptr + L]

            # Normality segment (normalized, sliced from concatenated vector)
            norm_seg_n = pix_norm_n_all[norm_ptr:norm_ptr + L]

            coef_seg = 1.0 - lam * norm_seg_n
            if clip_coef is not None:
                lo, hi = clip_coef
                coef_seg = np.clip(coef_seg, lo, hi)

            pix_final_all[pix_ptr:pix_ptr + L] = anom_seg * coef_seg

            pix_ptr += L
            norm_ptr += L

        # Overwrite pixel_preds
        self.pixel_preds[:] = list(pix_final_all.astype(np.float32))




    def calculate_metrics(self,path=None):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        if not path == None:
            numpy_save = normalize(self.image_preds)
            numpy_save = (numpy_save * 255).astype(np.uint8)
            numpy_save_gt = (self.image_labels*255).astype(np.uint8)[:,0]
            numpy_save = np.append(numpy_save, numpy_save_gt, axis=0)
            np.save(path, numpy_save)


        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        else:
            self.au_pro = 0



    # def run_coreset(self):
    #     # breakpoint()
    #     self.patch_lib = torch.cat(self.patch_lib, 0).cpu()
    #     n = len(self.patch_lib)
    #     self.f_coreset = 0.05
    #     if self.f_coreset < 1:
    #         # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
    #         #                                                 n=int(self.f_coreset * self.patch_lib.shape[0]),
    #         #                                                 eps=self.coreset_eps, )
    #         # Downsample memory bank to 1000 to try
    #         self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
    #                                                         n=1000,
    #                                                         eps=self.coreset_eps, )
    #         self.patch_lib = self.patch_lib[self.coreset_idx].to(self.args.device)

    # def run_coreset(self):
    #     # ===== normal memory bank =====
    #     self.patch_lib = torch.cat(self.patch_lib, 0).cpu()
    #     n = len(self.patch_lib)
    #     self.f_coreset = 0.05

    #     if self.f_coreset < 1:
    #         n_select = min(1000, self.patch_lib.shape[0])
    #         print("Downsampling normal library...")
    #         self.coreset_idx = self.get_coreset_idx_randomp(
    #             self.patch_lib,
    #             n=n_select,
    #             eps=self.coreset_eps,
    #         )
    #         self.patch_lib = self.patch_lib[self.coreset_idx].to(self.args.device)

    #     # ===== abnormal memory bank =====
    #     self.ab_patch_lib = torch.cat(self.ab_patch_lib, 0).cpu()
    #     n_ab = len(self.ab_patch_lib)

    #     if self.f_coreset < 1:
    #         n_select_ab = min(1000, self.ab_patch_lib.shape[0])
    #         print("Downsampling abnormal library...")
    #         self.ab_coreset_idx = self.get_coreset_idx_randomp(
    #             self.ab_patch_lib,
    #             n=n_select_ab,
    #             eps=self.coreset_eps,
    #         )
    #         self.ab_patch_lib = self.ab_patch_lib[self.ab_coreset_idx].to(self.args.device) 
            
    def run_coreset(self, use_ab_coreset: bool = False, ab_coreset_size: int = 1000):
        # ===== normal memory bank =====
        if isinstance(self.patch_lib, list):
            self.patch_lib = torch.cat(self.patch_lib, 0).cpu()
        self.patch_lib = self.patch_lib.cpu()
        self.f_coreset = 0.05

        if self.f_coreset < 1:
            n_select = min(1000, self.patch_lib.shape[0])
            print(f"[Coreset] Downsample normal library: {self.patch_lib.shape[0]} -> {n_select}")
            self.coreset_idx = self.get_coreset_idx_randomp(
                self.patch_lib,
                n=n_select,
                eps=self.coreset_eps,
            )
            self.patch_lib = self.patch_lib[self.coreset_idx].to(self.args.device)


    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)

