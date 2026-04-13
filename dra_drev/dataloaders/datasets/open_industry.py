import numpy as np
import os, sys
from dataloaders.datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms
import random
import glob
import open3d as o3d
import dataloaders.datasets.transform as aug_transform
from model.pointmae.patchcore.patchcore import PatchCore
from model.pointmae.feature_extractors.ransac_position import get_registration_np
import torch
from pathlib import Path

# Path segments in full paths used to derive parallel feature-cache trees.
OPEN_INDUSTRY_DATASET_MARKER = "Open-Industry"


class OpenIndustry(BaseADDataset):

    def __init__(self, args, train = True):
        super().__init__()
        self.args = args

        self.mask_num = 64

        self.SphereCropMask = aug_transform.SphereCropMask(part_num=self.mask_num)
        self.train_aug_compose = aug_transform.Compose([self.SphereCropMask])
        
        self.train = train
        self.classname = self.args.classname
        self.know_class = self.args.know_class
        self.pollution_rate = self.args.cont_rate
        self.device = args.device
        if self.args.test_threshold == 0 and self.args.test_rate == 0:
            self.test_threshold = self.args.nAnomaly
        else:
            self.test_threshold = self.args.test_threshold

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.transform_pcd = self.transform_pcd()
        self.transform_pcd_pseudo = self.transform_pcd_pseudo()
        if self.args.use_pseudo_anomaly:
            print("[OpenIndustry] pseudo anomaly enabled for training.")
        else:
            print("[OpenIndustry] pseudo anomaly disabled.")
        normal_data = list()
        split = 'train'

        normal_files = os.listdir(os.path.join(self.root, split))
        
        for file in normal_files:
            
            if 'pcd' in file[-3:]:
                normal_data.append(split +'/good/' +file)
        
        self.nPollution = int((len(normal_data)/(1-self.pollution_rate)) * self.pollution_rate)
        if self.test_threshold==0 and self.args.test_rate>0:
            self.test_threshold = int((len(normal_data)/(1-self.args.test_rate)) * self.args.test_rate) + self.args.nAnomaly


        self.ood_data = None
        # self.ood_data = self.get_ood_data()
        if self.train is False:
            normal_data = list()
            split = 'test'
            # all_test_files = os.path.join(self.root, split, '*.pcd')
            test_normal_pattern = os.path.join(self.root, split, f'{self.classname}_[0-9]*.pcd')
            test_normal_paths = sorted(glob.glob(test_normal_pattern))
            normal_files = [os.path.basename(p) for p in test_normal_paths]
            for file in normal_files:
                if 'pcd' in file[-3:]:
                    normal_data.append(split +'/good/'+ file)
        outlier_data, pollution_data = self.split_outlier()
        
        outlier_data.sort()
        
        normal_data = normal_data + pollution_data
        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        self.pcds = normal_data + outlier_data
        self.labels = np.array(normal_label + outlier_label)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()
        self.xyz_backbone = getattr(self.args, 'xyz_backbone', 'Point_MAE')
        print("[OpenIndustry] backbone:", self.xyz_backbone)
        if self.train:
            print(f"[OpenIndustry] train | class={self.classname} | total={len(self.pcds)}")
            print(f"  normal={len(self.normal_idx)} abnormal={len(self.outlier_idx)}")
            print(f"  know_class={self.args.know_class} nAnomaly(per class)={self.args.nAnomaly}")
        else:
            print(f"[OpenIndustry] test | class={self.classname} | total={len(self.pcds)}")
            print(f"  normal={len(self.normal_idx)} abnormal={len(self.outlier_idx)}")

    def _pcd_path_to_feature_cache_path(self, pcd_path: str) -> str:
        cache_root = f"OpenIndustry_{self.xyz_backbone}_feature"
        if OPEN_INDUSTRY_DATASET_MARKER in pcd_path:
            return pcd_path.replace(OPEN_INDUSTRY_DATASET_MARKER, cache_root, 1)
        return pcd_path

    def _init_feature_extractor(self):
        all_cache_exists = True
        print(f"Checking cache for backbone: {self.xyz_backbone}...")

        for index in range(len(self.pcds)):
            parts = os.path.join(self.root, self.pcds[index]).split('/')
            parts.pop(-2)
            pcd_path = '/'.join(parts)
            
            feature_path = self._pcd_path_to_feature_cache_path(pcd_path)
            feature_path = os.path.splitext(feature_path)[0] + ".npz"

            if not os.path.exists(feature_path):
                feature_path_pseudo = Path(feature_path)
                parts_pseudo = list(feature_path_pseudo.parts)
                parts_pseudo = ["pseudo" if p == "train" else p for p in parts_pseudo]
                feature_path_pseudo = str(Path(*parts_pseudo))
                
                if not os.path.exists(feature_path_pseudo):
                    all_cache_exists = False
                    break

        if all_cache_exists:
            print(f"[OpenIndustry] all {self.xyz_backbone} feature caches found.")
            print("[OpenIndustry] skipping backbone load (lower VRAM).")
            self.feature_extractor = None

            basic_template_path = os.listdir(os.path.join(self.args.dataset_root, self.args.classname, 'train'))[0]
            basic_template_path = os.path.join(self.args.dataset_root, self.args.classname, 'train',basic_template_path)
            pcd_o3d = o3d.io.read_point_cloud(basic_template_path)
            points = np.asarray(pcd_o3d.points).astype(np.float32)
            center = np.mean(points, axis=0, keepdims=True)
            points_centered = points - center
            scale = np.max(np.linalg.norm(points_centered, axis=1))
            points_norm = points_centered / (scale + 1e-6)
            self.basic_template = points_norm.astype(np.float32)
            return

        self.feature_extractor = PatchCore(self.device)
        basic_template_path = os.listdir(os.path.join(self.args.dataset_root, self.args.classname, 'train'))[0]
        basic_template_path = os.path.join(self.args.dataset_root, self.args.classname, 'train',basic_template_path)

        pcd_o3d = o3d.io.read_point_cloud(basic_template_path)

        points = np.asarray(pcd_o3d.points).astype(np.float32)

        center = np.mean(points, axis=0, keepdims=True)
        points_centered = points - center

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
            xyz_backbone_name = self.xyz_backbone
        )
        self.basic_template = basic_template
        self.feature_extractor.set_deep_feature_extractor()
        


    def split_outlier(self):
        outlier_data_dir = os.path.join(self.root, 'test')
        
        outlier_classes = ["Bump", "Deformation", "Dent", "Scar", "Scratch"]
        # outlier_classes = os.listdir(outlier_data_dir)
        if self.know_class is not None:
           
            know_outlier = []
            konw_outlier_test = []
            for know_class in self.know_class:
                assert know_class in outlier_classes, f"Known class '{know_class}' not in outlier classes {outlier_classes}"

                outlier_data = list()
                know_class_data = list()
                test_abnormal_pattern_sub = os.path.join(outlier_data_dir, f'{self.classname}_{know_class}*.pcd')
                test_abnormal_paths_sub = sorted(glob.glob(test_abnormal_pattern_sub))
                
                abnormal_files_sub = [os.path.basename(p) for p in test_abnormal_paths_sub]
                for file in abnormal_files_sub:
                    if 'pcd' in file[-3:]:
                        know_class_data.append('test/' + know_class + '/' + file)
                np.random.RandomState(self.args.ramdn_seed).shuffle(know_class_data)
                know_outlier.extend(know_class_data[0:self.args.nAnomaly])
                konw_outlier_test.extend(know_class_data[self.args.nAnomaly:])
                if self.train:
                    os.makedirs(self.args.experiment_dir, exist_ok=True)

                    with open(os.path.join(self.args.experiment_dir, 'abnormal_sample.txt'), 'a', encoding='utf-8') as w:
                        w.write("Abnormal samples moved from test to train:\n")
                        for item in know_outlier:
                            w.write(str(item) + "\n")
            for cl in outlier_classes:
                if cl == 'good':
                    continue
                test_abnormal_pattern_sub = os.path.join(outlier_data_dir, f'{self.classname}_{cl}*.pcd')
                test_abnormal_paths_sub = sorted(glob.glob(test_abnormal_pattern_sub))
                abnormal_files_sub = [os.path.basename(p) for p in test_abnormal_paths_sub]
                # outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
                if cl not in self.know_class:
                    for file in abnormal_files_sub:
                        if 'pcd' in file[-3:]:
                            outlier_data.append('test/' + cl + '/' + file)

            outlier_data.extend(konw_outlier_test)
            unknow_outlier = outlier_data
            if self.train:
                return know_outlier, list()
            else:
                
                return unknow_outlier, list()


    def load_pointcloud(self, path):
        parts = path.split('/')
        parts.pop(-2)
        path = '/'.join(parts)
        if path.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(path)
            if pcd.is_empty():
                raise ValueError(f"Point cloud is empty: {path}")
            return np.asarray(pcd.points)
        else:
            raise ValueError(f"Unsupported point cloud format: {path}")


    @staticmethod
    def preprocess_and_register(pc: np.ndarray, template: np.ndarray):
        """Center, scale to unit ball, then register ``pc`` to ``template``."""
        pc = np.asarray(pc, dtype=np.float32)

        center = np.mean(pc, axis=0, keepdims=True)
        pc = pc - center

        scale = np.max(np.linalg.norm(pc, axis=1))
        pc = pc / (scale + 1e-6)

        pc_reg = get_registration_np(pc, template)

        return pc_reg
    
    def transform_pcd(self):
        def _transform(pc):
            pc = self.preprocess_and_register(pc, self.basic_template).astype(np.float32)
            return pc

        return _transform

    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08):
        distances_to_center = np.linalg.norm(points - center, axis=1)
        max_distance = np.max(distances_to_center)
        movement_ratios = 1 - (distances_to_center / max_distance)
        movement_ratios = (movement_ratios - np.min(movement_ratios)) / (np.max(movement_ratios) - np.min(movement_ratios))

        directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])
        movements = movement_ratios * distance_to_move * directions
        new_points = points + np.abs(normals) * movements[:, np.newaxis]
        return new_points
    
    
    def transform_pcd_pseudo(self, knn=30):
        """Callable: pseudo anomaly (normalize/register + aug + local bump)."""

        def _transform(points, normals=None):
            points = np.asarray(points, dtype=np.float32)
            points = self.preprocess_and_register(points, self.basic_template).astype(np.float32)

            if normals is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
                )
                # pcd.orient_normals_consistent_tangent_plane(50)

                normals = np.asarray(pcd.normals, dtype=np.float32)
            else:
                normals = np.asarray(normals, dtype=np.float32)

            mask = np.ones(points.shape[0], dtype=np.int32) * -1

            Point_dict = {
                "coord": points,
                "normal": normals,
                "mask": mask
            }

            Point_dict, centers = self.train_aug_compose(Point_dict)

            xyz = Point_dict["coord"].astype(np.float32)
            normal = Point_dict["normal"].astype(np.float32)
            mask = Point_dict["mask"].astype(np.int32)

            mask[mask == (self.mask_num + 1)] = self.mask_num - 1

            num_shift = 1
            num_parts = len(centers)
            mask_range = np.arange(0, min(self.mask_num // 2, num_parts))

            shift_index = np.random.choice(mask_range, num_shift, replace=False)

            affected = np.isin(mask, shift_index)

            shift_xyz = xyz[affected].copy()
            shift_normal = normal[affected].copy()

            shifted_xyz_part = self.generate_pseudo_anomaly(
                shift_xyz,
                shift_normal,
                centers[shift_index[0]],
                distance_to_move=np.random.uniform(0.06, 0.12)
            )

            xyz_anom = xyz.copy()
            xyz_anom[affected] = shifted_xyz_part

            return xyz_anom.astype(np.float32)

        return _transform

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, index):
        rnd = random.randint(0, 2)
        if index in self.normal_idx and rnd == 0 and self.train and self.args.use_pseudo_anomaly:
            if self.ood_data is None:
                index = random.choice(self.normal_idx)
                pcd = self.load_pointcloud(os.path.join(self.root, self.pcds[index]))

                transform = self.transform_pcd_pseudo
            label = 2
        else:
            pcd = self.load_pointcloud(os.path.join(self.root, self.pcds[index]))
            transform = self.transform_pcd
            label = self.labels[index]
            if not self.train and label == 1:
               
                if self.pcds[index].split('/')[1] not in self.know_class:
                    label = 2
                    
        parts = os.path.join(self.root, self.pcds[index])
        parts = parts.split('/')
        parts.pop(-2)
        pcd_path = '/'.join(parts)
        
        feature_path = self._pcd_path_to_feature_cache_path(pcd_path)
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        feature_path = os.path.splitext(feature_path)[0] + ".npz"
        if label == 2:
            feature_path = Path(feature_path)
            parts = list(feature_path.parts)
            parts = ["pseudo" if p == "train" else p for p in parts]
            feature_path = Path(*parts)
            feature_path = str(feature_path)

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        
        if os.path.exists(feature_path):
            data = np.load(feature_path, allow_pickle=True)
            pmae_features = data["feat"]               # Tensor
            center_idx    = data.get("center_idx", None)
            sample = {
                'pcd_features': pmae_features,
                'label': label
            }
            return sample
        else:
            if self.feature_extractor is None:
                raise RuntimeError(
                    f"Feature extractor is None (cache-only mode) but cache is missing: {feature_path}\n"
                    f"Restore cache files or delete partial caches and rerun to load the backbone."
                )

            pcd = transform(pcd)
            pcd = torch.from_numpy(pcd).float().to(self.device)
            assert pcd is not None and len(pcd) > 0, \
                f"[ERROR] Empty point cloud found! Path: {os.path.join(self.root, self.pcds[index])}"
                
            with torch.no_grad():
                pmae_features, center_idx = self.feature_extractor._embed_pointmae(pcd)  
            center_idx = center_idx.detach().cpu().numpy()
            
            np.savez(feature_path, feat=pmae_features, center_idx=center_idx)
            print(f"[SAVE] feature saved to: {feature_path}")
            print(f"pcd_features->Label: {int(label)}, Mean: {pmae_features.mean().item():.6f}, Var: {pmae_features.var().item():.6f}")
            # torch.cuda.empty_cache()
            sample = {'pcd_features': pmae_features, 'label': label}
            return sample


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace

    args = SimpleNamespace(
        classname="fangxiedianpian",
        know_class=["Bump", "Deformation"],
        cont_rate=0.1,
        test_threshold=0,
        test_rate=0.0,
        nAnomaly=5,
        dataset_root="/path/to/dataset",
        ramdn_seed=42,
        outlier_root=None,
        xyz_backbone="Point_MAE",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_pseudo_anomaly=1,
        experiment_dir="./tmp_dataset_debug",
    )
    test_ds = OpenIndustry(args, train=False)
    test_ds._init_feature_extractor()
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    print("[open_industry smoke test] keys:", list(batch.keys()))
    print("[open_industry smoke test] label:", batch["label"])