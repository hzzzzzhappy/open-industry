from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from feature_extractors.features import *
# from feature_extractors.models import *
from feature_extractors.pointnet2_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm


only_use_points = True


from data.real3d import voxel_size_setting
from feature_extractors.shape_context import get_shape_context 
from feature_extractors.CVFH import get_CVFH 
from feature_extractors.NARF import get_NARF 
from feature_extractors.Spin import get_Spin 
from feature_extractors.Unique_shape import get_USC 
from feature_extractors.SHOT import get_SHOT
import data.datasets.transform as aug_transform
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'






def batched_knn(knn,reference, query, batch_size=4000):
    all_idx = []
    for i in range(0, query.shape[1], batch_size):
        q_batch = query[:, i:i+batch_size, :]  # shape [B, b, 3]
        _, idx = knn(reference, q_batch)    # shape [B, b, k]
        all_idx.append(idx)
    return torch.cat(all_idx, dim=1)           # [B, G, k]





class FPFHFeatures(Features):

    def __init__(self,args=None):
        self.args = args
        super().__init__(args = args)
        self.mask_num = 64
        self.SphereCropMask = aug_transform.SphereCropMask(part_num=self.mask_num)
        self.train_aug_compose = aug_transform.Compose([self.SphereCropMask])

    def get_fpfh_features(self,unorganized_pc, voxel_size=0.1,mask = None ,abnormal_ratio_threshold = 0.4 ,test = False):
        # unorganized_pc:(1,n,3)--> (n,3)
        unorganized_pc = unorganized_pc.squeeze(0).numpy() #(n,3)


        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
            unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        else:
            unorganized_pc_no_zeros = unorganized_pc
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

        voxel_size = 1000000
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10000, max_nn=10))

        radius_feature = 1000000
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=self.args.max_nn))
        fpfh_np = np.asarray(pcd_fpfh.data.T, dtype=np.float32)
        fpfh = torch.from_numpy(fpfh_np).to(self.args.device)


        if self.args.use_MSND:
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
            (radius=radius_feature, max_nn=2*self.args.max_nn))
            # fpfh2 = torch.Tensor(pcd_fpfh.data.T).cuda()
            fpfh2 = torch.from_numpy(pcd_fpfh.data.T.copy()).float().cuda()
            fpfh = torch.cat([fpfh,fpfh2],dim=-1)


            if self.args.num_MSND == 2:
                pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
                (radius=radius_feature, max_nn=3*self.args.max_nn))
                # fpfh3 = torch.Tensor(pcd_fpfh.data.T).cuda()
                fpfh3 = torch.from_numpy(pcd_fpfh.data.T.copy()).float().cuda()

                fpfh = torch.cat([fpfh,fpfh2,fpfh3],dim=-1)
        
        # fps the centers out
        # unorganized_pc_no_zeros expected (1,n,3)
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc_no_zeros).cuda().unsqueeze(dim=0)
        unorganized_pc_no_zeros = unorganized_pc_no_zeros.to(torch.float32)
        # unorganized_pc_no_zeros expected (1,n,3)
        batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
        # center -> [1, 4096, 3], center_idx -> [1, 4096] Should use center_idx and unorganized_pc_no_zeros
        center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), self.args.num_group)  # B G 3
        # knn to get the neighborhood
        knn = KNN(k=self.args.group_size, transpose_mode=True)

        idx = batched_knn(knn,unorganized_pc_no_zeros, center)  # B G M
        #ori_idx -> [1, 4096, 128]
        ori_idx = idx

        idx_base = torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device).view(-1, 1, 1) * num_points
        
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = fpfh.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.args.num_group, self.args.group_size, -1).contiguous()
        # print("neighborhood",neighborhood.shape)
        agg_point_feature = torch.mean(neighborhood,-2)
        if self.args.use_LFSA:
            # agg_point_feature -> [4096, 132]
            agg_point_feature = agg_point_feature.squeeze()
        else:
            agg_point_feature = fpfh.squeeze()
        unorganized_pc = torch.tensor(unorganized_pc)

        # agg_point_feature (group,f)
        # unorganized_pc (n,3)
        # unorganized_pc_no_zeros (1,n,3)
        # center (1,group,3)
        # Separate abnormal patches and normal patches during training
        # mask -> [1, 1, 188184]
        if test:
            return agg_point_feature, unorganized_pc, unorganized_pc_no_zeros, center, _
        if mask is not None:
            abnormal_ratio_threshold = abnormal_ratio_threshold  # Abnormal patch ratio threshold
            device = ori_idx.device

            # mask: [B,1,N] -> [B,N] bool
            mask_bool = (mask.to(device).squeeze(1) > 0)  # True=abnormal point

            B, G, M = ori_idx.shape
            patch_point_mask = mask_bool.gather(1, ori_idx.view(B, -1)).view(B, G, M)  # [B,G,M]

            # Abnormal point count/ratio for each patch
            abn_cnt   = patch_point_mask.sum(dim=-1)              # [B,G]  Abnormal point count
            abn_ratio = abn_cnt.float() / float(M)                # [B,G]  Abnormal point ratio

            # Normal patch: cannot have any abnormal points
            patch_nor_mask = (abn_cnt == 0)                       # [B,G] bool

            # Abnormal patch: abnormal ratio reaches threshold
            patch_abn_mask = (abn_ratio > abnormal_ratio_threshold)  # [B,G] bool

            # Optional: uncertain patch: has abnormalities but ratio not enough, not added to any library
            patch_ign_mask = (~patch_nor_mask) & (~patch_abn_mask)    # [B,G] bool

            # Convert to idx (using batch 0)
            nor_idx = patch_nor_mask[0]
            abn_idx = patch_abn_mask[0]
            ign_idx = patch_ign_mask[0]

            agg_feat_nor = agg_point_feature[nor_idx]   # [G_nor, 132]
            agg_feat_abn = agg_point_feature[abn_idx]   # [G_abn, 132]

            print(f"normal patches (0 abn): {nor_idx.sum().item()}, "
                f"abnormal patches (>=thr): {abn_idx.sum().item()}, "
                f"ignored (in-between): {ign_idx.sum().item()}")

            return agg_feat_nor, unorganized_pc, unorganized_pc_no_zeros, center, agg_feat_abn
        else:
            # Normal point cloud: use calculated agg_point_feature directly without any additional filtering
            agg_feat_nor = agg_point_feature  # [G, F] (recommend use_LFSA=True for patch-level)

            # 1) Generate pseudo-abnormal point cloud + pseudo mask (mask >0 indicates abnormal point)
            pcd_pseudo, pseudo_mask = self.transform_pcd_pseudo(unorganized_pc)

            device = self.args.device

            # ---- Shape unification: pcd_pseudo -> [1,N,3], pseudo_mask -> [1,1,N]
            if isinstance(pcd_pseudo, np.ndarray):
                pcd_pseudo = torch.from_numpy(pcd_pseudo)
            if pcd_pseudo.dim() == 2:
                pcd_pseudo = pcd_pseudo.unsqueeze(0)
            pcd_pseudo = pcd_pseudo.to(device=device, dtype=torch.float32)

            if isinstance(pseudo_mask, np.ndarray):
                pseudo_mask = torch.from_numpy(pseudo_mask)
            if pseudo_mask.dim() == 1:
                pseudo_mask = pseudo_mask.view(1, 1, -1)
            elif pseudo_mask.dim() == 2:
                pseudo_mask = pseudo_mask.unsqueeze(1)
            pseudo_mask = pseudo_mask.to(device=device)

            # 2) Pseudo-abnormal point cloud: compute FPFH (consistent with above)
            pcd_pseudo_np = pcd_pseudo.squeeze(0).detach().cpu().numpy()

            pcd_pseudo_np_no0 = pcd_pseudo_np

            o3d_pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_pseudo_np_no0))
            o3d_pc2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10000, max_nn=10))

            radius_feature2 = 1000000
            pcd_fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
                o3d_pc2,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature2, max_nn=self.args.max_nn)
            )
            fpfh_pseudo = torch.from_numpy(np.asarray(pcd_fpfh2.data.T, dtype=np.float32)).to(device)

            if self.args.use_MSND:
                pcd_fpfh2b = o3d.pipelines.registration.compute_fpfh_feature(
                    o3d_pc2,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature2, max_nn=2 * self.args.max_nn)
                )
                fpfh2b = torch.from_numpy(pcd_fpfh2b.data.T.copy()).float().to(device)
                fpfh_pseudo = torch.cat([fpfh_pseudo, fpfh2b], dim=-1)

                if self.args.num_MSND == 2:
                    pcd_fpfh2c = o3d.pipelines.registration.compute_fpfh_feature(
                        o3d_pc2,
                        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature2, max_nn=3 * self.args.max_nn)
                    )
                    fpfh2c = torch.from_numpy(pcd_fpfh2c.data.T.copy()).float().to(device)
                    fpfh_pseudo = torch.cat([fpfh_pseudo, fpfh2b, fpfh2c], dim=-1)

            # 3) Pseudo-abnormal point cloud: FPS + KNN aggregate to patch features (consistent with above)
            pc_no0 = torch.tensor(pcd_pseudo_np_no0, device=device, dtype=torch.float32).unsqueeze(0)  # [1,N,3]
            B, N, _ = pc_no0.shape

            center2, center_idx2 = fps(pc_no0.contiguous(), self.args.num_group)  # [1,G,3],[1,G]
            knn2 = KNN(k=self.args.group_size, transpose_mode=True)
            ori_idx2 = batched_knn(knn2, pc_no0, center2)  # [1,G,M]

            idx_base2 = torch.arange(0, B, device=device).view(-1, 1, 1) * N
            flat_idx2 = (ori_idx2 + idx_base2).view(-1)

            neighborhood2 = fpfh_pseudo.reshape(B * N, -1)[flat_idx2, :]
            neighborhood2 = neighborhood2.view(B, self.args.num_group, self.args.group_size, -1).contiguous()
            agg_point_feature2 = torch.mean(neighborhood2, -2).squeeze(0)  # [G,F]

            # 4) Use pseudo_mask to slice abnormal patches (>= abnormal_ratio_threshold)
            mask_bool2 = (pseudo_mask.squeeze(1) > 0)  # [1,N] bool
            B, G, M = ori_idx2.shape

            patch_point_mask2 = mask_bool2.gather(1, ori_idx2.view(B, -1)).view(B, G, M)  # [1,G,M]
            abn_cnt2 = patch_point_mask2.sum(dim=-1)                 # [1,G]
            abn_ratio2 = abn_cnt2.float() / float(M)                 # [1,G]

            patch_abn_mask2 = (abn_ratio2 > abnormal_ratio_threshold)[0]  # [G]
            agg_feat_abn = agg_point_feature2[patch_abn_mask2]             # [G_abn, F]

            print(f"[Pseudo] abnormal patches added: {agg_feat_abn.shape[0]} (thr={abnormal_ratio_threshold})")

            return agg_feat_nor, unorganized_pc, unorganized_pc_no_zeros, center, agg_feat_abn
        # print(agg_point_feature.shape,unorganized_pc.shape,unorganized_pc_no_zeros.shape,center.shape)
        return agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center

    def get_features(self,unorganized_pc ,mask = None , abnormal_ratio_threshold = 0.4 ,test = False):
        if self.args.feature == 'FPFH':
            return self.get_fpfh_features(unorganized_pc ,mask = mask ,abnormal_ratio_threshold = abnormal_ratio_threshold ,test = test)
        if self.args.feature == 'shape_context':
            return get_shape_context(unorganized_pc)
        if self.args.feature == 'CVFH':
            return get_CVFH(unorganized_pc)
        if self.args.feature == 'NARF':
            return get_NARF(unorganized_pc)
        if self.args.feature == 'Spin':
            return get_Spin(unorganized_pc)
        if self.args.feature == 'USC':
            return get_USC(unorganized_pc)
        if self.args.feature == 'SHOT':
            return get_SHOT(unorganized_pc)  





    def collect_features(self,pc ,mask = None ,abnormal_ratio_threshold = 0.4):
        # pc:(1,n,3)

        # agg_point_feature (group,33)
        # unorganized_pc (n,3)
        # unorganized_pc_no_zeros (1,n,3)
        # center (1,group,3)
        # Need to know the idx of center points selected by FPS, and the idx of all points gathered by KNN for each idx
        # Pass mask during training, do not pass mask during testing
        if (mask.squeeze(1) > 0).sum().item() != 0:
            feature_maps,_,_,_,ab_feature_maps= self.get_features(pc,mask = mask ,abnormal_ratio_threshold = abnormal_ratio_threshold)
            self.patch_lib.append(feature_maps)
            self.ab_patch_lib.append(ab_feature_maps)
        else:
            feature_maps,_,_,_,ab_feature_maps = self.get_features(pc,abnormal_ratio_threshold = abnormal_ratio_threshold)
            self.patch_lib.append(feature_maps)
            self.ab_patch_lib.append(ab_feature_maps)
        # Use one here too

    def predict(self, pc, mask, label, path=None):
        # agg_point_feature (group,33)
        # unorganized_pc (n,3)
        # unorganized_pc_no_zeros (1,n,3)
        # center (1,group,3)

        agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center,_ = self.get_features(pc,test=True)
        self.compute_anomay_scores(agg_point_feature, mask, label,path, unorganized_pc,unorganized_pc_no_zeros,center)
        # Compute distance to abnormal patch library, then subtract to get normality score, finally subtract from original normality score
        self.compute_normality_scores(agg_point_feature, mask, label,path, unorganized_pc,unorganized_pc_no_zeros,center)
        # I want to get the calculated pred values here, then normalize the two groups of pred values and subtract them


    def transform_pcd_pseudo(
        self,
        points,
        normals=None,
        knn=30,
    ):
        """
        Input:
            points:  (N,3) Original point cloud (numpy / Tensor)
            normals: (N,3) Normal vectors, can be None
        Output:
            xyz_anom:    (N,3) Pseudo-abnormal point cloud
            pseudo_mask: (N,)  0/1, abnormal=1, normal=0
        """

        # -------- Convert to numpy uniformly --------
        points = np.asarray(points, dtype=np.float32)

        # -------- normals --------
        if normals is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
            )
            normals = np.asarray(pcd.normals, dtype=np.float32)
        else:
            normals = np.asarray(normals, dtype=np.float32)

        # -------- Initialize mask (for SphereCropMask) --------
        mask = np.ones(points.shape[0], dtype=np.int32) * -1

        Point_dict = {
            "coord": points,
            "normal": normals,
            "mask": mask
        }

        # -------- Train augment (will generate patch mask + centers) --------
        Point_dict, centers = self.train_aug_compose(Point_dict)

        xyz = Point_dict["coord"].astype(np.float32)
        normal = Point_dict["normal"].astype(np.float32)
        mask = Point_dict["mask"].astype(np.int32)

        mask[mask == (self.mask_num + 1)] = self.mask_num - 1

        # -------- Randomly select 1 patch for pseudo-abnormality --------
        num_parts = len(centers)
        mask_range = np.arange(0, min(self.mask_num // 2, num_parts))
        shift_index = np.random.choice(mask_range, 1, replace=False)

        affected = np.isin(mask, shift_index)   # (N,) bool

        shift_xyz = xyz[affected].copy()
        shift_normal = normal[affected].copy()


        def generate_pseudo_anomaly( points, normals, center, distance_to_move=0.08):
            distances_to_center = np.linalg.norm(points - center, axis=1)
            max_distance = np.max(distances_to_center)
            movement_ratios = 1 - (distances_to_center / max_distance)
            movement_ratios = (movement_ratios - np.min(movement_ratios)) / (np.max(movement_ratios) - np.min(movement_ratios))

            directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])
            movements = movement_ratios * distance_to_move * directions
            new_points = points + np.abs(normals) * movements[:, np.newaxis]
            return new_points
        shifted_xyz_part = generate_pseudo_anomaly(
            shift_xyz,
            shift_normal,
            centers[shift_index[0]],
            distance_to_move=np.random.uniform(0.06, 0.12)
        )

        # -------- Fill back into whole point cloud --------
        xyz_anom = xyz.copy()
        xyz_anom[affected] = shifted_xyz_part

        # -------- 0/1 mask: abnormal=1, normal=0 --------
        pseudo_mask = affected.astype(np.int64)

        return xyz_anom.astype(np.float32), pseudo_mask



 