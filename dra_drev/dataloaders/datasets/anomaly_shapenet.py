import numpy as np
import os
import sys
from dataloaders.datasets.base_dataset import BaseADDataset
import open3d as o3d
import glob
import random
import torch
from pathlib import Path
from model.pointmae.patchcore.patchcore import PatchCore
from model.pointmae.feature_extractors.ransac_position import get_registration_np
import dataloaders.datasets.transform as aug_transform


class AnomalyShapeNet(BaseADDataset):
    """
    Anomaly-ShapeNet-v2 / Real3D-AD dataset loader.

    Directory layout:
        classname/
        ├── train/          # Normal samples (.pcd)
        ├── test/
        │   └── good/       # Test normal samples (.pcd)
        └── GT/             # Anomaly annotations (.txt)
            ├── bulge/
            ├── broken/
            ├── concavity/
            ├── crak/
            └── scratch/
    """

    _shared_feature_extractor = None
    _shared_backbone_name = None
    _shared_device = None

    def __init__(self, args, train=True):
        super(AnomalyShapeNet, self).__init__()
        self.args = args
        self.train = train
        self.classname = self.args.classname
        self.know_class = self.args.know_class
        self.nAnomaly = self.args.nAnomaly
        self.device = args.device

        self.mask_num = 64
        self.SphereCropMask = aug_transform.SphereCropMask(part_num=self.mask_num)
        self.train_aug_compose = aug_transform.Compose([self.SphereCropMask])

        use_pseudo = getattr(self.args, 'use_pseudo_anomaly', False)
        if use_pseudo:
            print("✔ Pseudo anomaly generation enabled for training.")
        else:
            print("✔ Pseudo anomaly generation disabled.")

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.all_anomaly_classes = ["bulge", "broken", "concavity", "crak", "scratch"]
        self.xyz_backbone = getattr(self.args, 'xyz_backbone', 'Point_BERT')
        print(f"Feature extractor: {self.xyz_backbone}")
        self.feature_extractor = None
        normal_data = self._load_normal_data()
        self._init_basic_template()

        outlier_data, pollution_data = self._split_outlier()

        outlier_data.sort()
        normal_data = normal_data + pollution_data

        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        self.pcds = normal_data + outlier_data
        self.labels = np.array(normal_label + outlier_label)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()

        self.xyz_backbone = getattr(self.args, 'xyz_backbone', 'Point_BERT')

        if self.train:
            print(f"🔥Training set loading... -> {self.classname}")
            print(f"Training set total samples: {len(self.pcds)}")
            print(f"Training set normal: {len(self.normal_idx)}, anomaly: {len(self.outlier_idx)}")
            print(f"Known anomaly classes: {self.args.know_class}, samples per class: {self.args.nAnomaly}")
        else:
            print(f"🔍Test set loading... -> {self.classname}")
            print(f"Test set total samples: {len(self.pcds)}")
            print(f"Test set normal: {len(self.normal_idx)}, anomaly: {len(self.outlier_idx)}")

    def _load_normal_data(self):
        """Load normal sample data"""
        normal_data = []

        train_dir = os.path.join(self.root, 'train')
        if os.path.exists(train_dir):
            pcd_files = glob.glob(os.path.join(train_dir, '*.pcd'))
            for pcd_file in pcd_files:
                rel_path = os.path.relpath(pcd_file, self.root)
                normal_data.append(rel_path)

        train_good_count = len(normal_data)

        test_good_count = 0
        if not self.train:
            normal_data = []

            test_dir = os.path.join(self.root, 'test')
            if os.path.exists(test_dir):
                good_dir = os.path.join(test_dir, 'good')
                if os.path.exists(good_dir):
                    pcd_files = glob.glob(os.path.join(good_dir, '*.pcd'))
                    test_good_count = len(pcd_files)
                    for pcd_file in pcd_files:
                        rel_path = os.path.relpath(pcd_file, self.root)
                        normal_data.append(rel_path)
                else:
                    pcd_files = glob.glob(os.path.join(test_dir, '*_good*.pcd'))
                    pcd_files += glob.glob(os.path.join(test_dir, '*_positive*.pcd'))
                    test_good_count = len(pcd_files)
                    for pcd_file in pcd_files:
                        rel_path = os.path.relpath(pcd_file, self.root)
                        normal_data.append(rel_path)

                        print(f"Loaded normal samples: train/good={train_good_count}, test/good={test_good_count}, total={len(normal_data)}")
        return normal_data

    def _init_basic_template(self):
        """Initialize basic_template (for point cloud registration)"""
        train_dir = os.path.join(self.root, 'train')
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")

        pcd_files = glob.glob(os.path.join(train_dir, '*.pcd'))
        if len(pcd_files) == 0:
            raise ValueError(f"No pcd files in training directory: {train_dir}")

        basic_template_path = pcd_files[0]
        pcd_o3d = o3d.io.read_point_cloud(basic_template_path)
        points = np.asarray(pcd_o3d.points).astype(np.float32)
        center = np.mean(points, axis=0, keepdims=True)
        points_centered = points - center
        scale = np.max(np.linalg.norm(points_centered, axis=1))
        points_norm = points_centered / (scale + 1e-6)

        self.basic_template = points_norm.astype(np.float32)
        print(f"basic_template initialized, using file: {os.path.basename(basic_template_path)}")

    def _split_outlier(self):
        """Split anomaly data: known anomalies to training set, rest to test set"""
        gt_dir_capital = os.path.join(self.root, 'GT')
        gt_dir_lower = os.path.join(self.root, 'gt')

        gt_dir = None
        gt_dir_name = None
        if os.path.exists(gt_dir_capital):
            gt_dir = gt_dir_capital
            gt_dir_name = 'GT'
        elif os.path.exists(gt_dir_lower):
            gt_dir = gt_dir_lower
            gt_dir_name = 'gt'
        else:
            print(f"[WARNING] GT or gt directory not found")
            return [], []

        test_dir = os.path.join(self.root, 'test')
        if not os.path.exists(test_dir):
            print(f"[WARNING] test directory not found")
            return [], []

        know_outlier = []
        unknow_outlier = []
        all_txt_files = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))
        anomaly_groups = {}
        for txt_file in all_txt_files:
            basename = os.path.basename(txt_file)
            name_without_ext = os.path.splitext(basename)[0]

            prefix = self.classname + '_'
            if name_without_ext.startswith(prefix):
                remainder = name_without_ext[len(prefix):]
                anomaly_type = ''.join([c for c in remainder if not c.isdigit()])
            else:
                parts = name_without_ext.split('_')
                if len(parts) >= 2:
                    anomaly_type = parts[-1]
                    anomaly_type = ''.join([c for c in anomaly_type if c.isalpha()])
                else:
                    continue

            if anomaly_type not in anomaly_groups:
                anomaly_groups[anomaly_type] = []
            anomaly_groups[anomaly_type].append(txt_file)

            print(f"Anomaly types found in GT/ directory: {list(anomaly_groups.keys())}")
        for atype, files in anomaly_groups.items():
            print(f"  {atype}: {len(files)} files")

        if self.know_class is not None and len(anomaly_groups) > 0:
            print(f"Known anomaly class config: {self.know_class}")
            for know_class in self.know_class:
                if know_class in anomaly_groups:
                    txt_files = anomaly_groups[know_class]
                    np.random.RandomState(self.args.ramdn_seed).shuffle(txt_files)
                    for i, txt_file in enumerate(txt_files):
                        rel_path = os.path.join(gt_dir_name, os.path.basename(txt_file))
                        if i < self.args.nAnomaly:
                            know_outlier.append(rel_path)
                        else:
                            unknow_outlier.append(rel_path)

                    test_count = max(0, len(txt_files) - self.args.nAnomaly)
                    print(f"  {know_class}: total {len(txt_files)} ({gt_dir_name}/), train {min(self.args.nAnomaly, len(txt_files))}, test {test_count}")

        print(f"Unknown anomaly classes ({gt_dir_name}/):")
        for anomaly_class in anomaly_groups.keys():
            if anomaly_class not in self.know_class:
                txt_files = anomaly_groups[anomaly_class]
                for txt_file in txt_files:
                    rel_path = os.path.join(gt_dir_name, os.path.basename(txt_file))
                    unknow_outlier.append(rel_path)
                    print(f"  {anomaly_class}: all {len(txt_files)} files go to test set")

        if self.train:
            return know_outlier, []
        else:
            return unknow_outlier, []

    def load_pointcloud(self, path):
        """
        Load a point cloud file (.pcd or .txt format).

        Args:
            path: path relative to self.root

        Returns:
            numpy.ndarray of shape (N, 3)
        """
        full_path = os.path.join(self.root, path)

        if full_path.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(full_path)
            if pcd.is_empty():
                raise ValueError(f"Point cloud file is empty: {full_path}")
            return np.asarray(pcd.points)

        elif full_path.endswith('.txt'):
            try:
                data = np.loadtxt(full_path, delimiter=',')
            except:
                data = np.loadtxt(full_path, delimiter=' ')

            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data[:, :3].astype(np.float32)

        else:
            raise ValueError(f"Unsupported file format: {full_path}")

    @staticmethod
    def preprocess_and_register(pc: np.ndarray, template: np.ndarray):
        """
        Preprocess and register a point cloud to the template.

        Args:
            pc: (N, 3) raw point cloud
            template: (M, 3) template point cloud

        Returns:
            (N, 3) registered point cloud
        """
        pc = np.asarray(pc, dtype=np.float32)
        center = np.mean(pc, axis=0, keepdims=True)
        pc = pc - center
        scale = np.max(np.linalg.norm(pc, axis=1))
        pc = pc / (scale + 1e-6)
        pc_reg = get_registration_np(pc, template)

        return pc_reg

    def transform_pcd(self):
        """Return point cloud preprocessing function"""
        def _transform(pc):
            pc = self.preprocess_and_register(pc, self.basic_template).astype(np.float32)
            return pc

        return _transform

    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08):
        """
        Generate a local geometric pseudo-anomaly (bump/dent).

        Args:
            points: (N, 3) local patch coordinates
            normals: (N, 3) surface normals
            center: (3,) patch centre
            distance_to_move: controls anomaly magnitude

        Returns:
            (N, 3) deformed point cloud
        """
        distances_to_center = np.linalg.norm(points - center, axis=1)
        max_distance = np.max(distances_to_center)
        movement_ratios = 1 - (distances_to_center / max_distance)
        movement_ratios = (movement_ratios - np.min(movement_ratios)) / \
                          (np.max(movement_ratios) - np.min(movement_ratios))

        directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])
        movements = movement_ratios * distance_to_move * directions
        new_points = points + np.abs(normals) * movements[:, np.newaxis]
        return new_points

    def transform_pcd_pseudo(self, knn=30):
        """
        Return a callable that generates a pseudo-anomaly point cloud.

        Args:
            knn: K for normal estimation

        Returns:
            transform function
        """
        def _transform(points, normals=None):
            points = np.asarray(points, dtype=np.float32)
            points = self.preprocess_and_register(points, self.basic_template).astype(np.float32)

            if normals is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
                )

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
        use_pseudo = getattr(self.args, 'use_pseudo_anomaly', False)
        rnd = random.randint(0, 2)

        if index in self.normal_idx and rnd == 0 and self.train and use_pseudo:
            pseudo_index = random.choice(self.normal_idx)
            pcd_path = self.pcds[pseudo_index]
            pcd = self.load_pointcloud(os.path.join(self.root, pcd_path))
            transform = self.transform_pcd_pseudo()
            pcd = transform(pcd)
            label = 2
        else:
            pcd_path = self.pcds[index]
            pcd = self.load_pointcloud(pcd_path)
            transform = self.transform_pcd()
            pcd = transform(pcd)
            label = self.labels[index]

        if not self.train and label == 1:
            anomaly_type = self._get_anomaly_type_from_path(pcd_path)
            if anomaly_type not in self.know_class:
                label = 2

        pcd_np = pcd.copy()
        pcd = torch.from_numpy(pcd).float().to(self.device)
        feature_path = self._get_feature_cache_path(pcd_path, label)

        if os.path.exists(feature_path):
            data = np.load(feature_path, allow_pickle=True)
            pmae_features = data["feat"]
            center_idx = data.get("center_idx", None)
        else:
            if self.feature_extractor is None:
                raise RuntimeError("Feature extractor not initialized, please call _init_feature_extractor()")
            with torch.no_grad():
                pmae_features, center_idx = self.feature_extractor._embed_pointmae(pcd)

            center_idx = center_idx.detach().cpu().numpy()
            np.savez(feature_path, feat=pmae_features, center_idx=center_idx)
            print(f"[SAVE] feature saved: {feature_path}")

        if label == 0:
            points_label = np.zeros(len(pcd_np), dtype=np.int64)
        else:
            points_label = self._load_point_labels(pcd_path, label)

        sample = {
            'pcd_features': pmae_features,
            'label': label,
            'coord': pcd_np,
            'points_label': points_label
        }

        return sample

    def _get_anomaly_type_from_path(self, path):
        """Extract anomaly type from path"""
        basename = os.path.basename(path)
        name_without_ext = os.path.splitext(basename)[0]

        prefix = self.classname + '_'
        if name_without_ext.startswith(prefix):
            remainder = name_without_ext[len(prefix):]
            anomaly_type = ''.join([c for c in remainder if not c.isdigit()])
            return anomaly_type

        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            anomaly_type = parts[-1]
            anomaly_type = ''.join([c for c in anomaly_type if not c.isdigit()])
            return anomaly_type

        return None

    def _get_feature_cache_path(self, pcd_path, label):
        """Generate feature cache file path"""
        full_path = os.path.join(self.root, pcd_path)
        cache_dir = f"/root/autodl-tmp/{self.xyz_backbone}_features"
        feature_path = full_path.replace(self.args.dataset_root, cache_dir)
        feature_path = os.path.splitext(feature_path)[0] + ".npz"

        if label == 2:
            feature_path = Path(feature_path)
            parts = list(feature_path.parts)
            parts = ["pseudo" if p == "train" else p for p in parts]
            feature_path = Path(*parts)
            feature_path = str(feature_path)

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)

        return feature_path

    def _load_point_labels(self, pcd_path, label):
        """
        Load per-point anomaly labels.

        Args:
            pcd_path: point cloud path relative to dataset root
            label: sample-level label (0=normal, 1=seen anomaly, 2=unseen anomaly)

        Returns:
            numpy.ndarray of shape (N,), 0=normal, 1=anomaly
        """
        if label == 0:
            return None

        if 'template' in pcd_path.lower():
            return np.zeros(2048, dtype=np.int64)

        full_pcd_path = os.path.join(self.root, pcd_path)
        basename = os.path.splitext(os.path.basename(full_pcd_path))[0]
        gt_path = os.path.join(self.root, "gt", f"{basename}.txt")

        if not os.path.exists(gt_path):
            print(f"[Warning] GT file not found: {gt_path}, returning all anomalies")
            return np.ones(2048, dtype=np.int64)

        try:
            gt_data = np.loadtxt(gt_path, delimiter=' ')
            if gt_data.ndim == 1:
                gt_data = gt_data.reshape(1, -1)
            point_labels = gt_data[:, -1].astype(np.int64)
            point_labels = (point_labels > 0).astype(np.int64)
            return point_labels
        except Exception as e:
            print(f"[Error] Failed to load GT labels from {gt_path}: {e}")
            return np.ones(2048, dtype=np.int64)

    def _init_feature_extractor(self):
        """Initialize feature extractor (shared via class variable to avoid repeated loading)"""
        from model.pointmae.patchcore.patchcore import PatchCore
        from model.pointmae.feature_extractors.ransac_position import get_registration_np

        if (AnomalyShapeNet._shared_feature_extractor is not None and
            AnomalyShapeNet._shared_backbone_name == self.xyz_backbone and
            AnomalyShapeNet._shared_device == self.device):
            self.feature_extractor = AnomalyShapeNet._shared_feature_extractor
            print(f"✔ Reusing existing {self.xyz_backbone} feature extractor (train={self.train}, samples={len(self.pcds)})")

            all_cache_exists = True
            for index in range(len(self.pcds)):
                pcd_path = self.pcds[index]
                label = self.labels[index]

                if not self.train and label == 1:
                    anomaly_type = self._get_anomaly_type_from_path(pcd_path)
                    if anomaly_type not in self.know_class:
                        label = 2

                feature_path = self._get_feature_cache_path(pcd_path, label)

                if not os.path.exists(feature_path):
                    all_cache_exists = False
                    print(f"[WARN] Cache missing for {pcd_path}, will create new feature extractor")
                    break

            if all_cache_exists:
                return

            print(f"[INFO] Recreating feature extractor due to missing cache")

        all_cache_exists = True
        print(f"Checking cache for backbone: {self.xyz_backbone}...")

        for index in range(len(self.pcds)):
            pcd_path = self.pcds[index]
            label = self.labels[index]

            if not self.train and label == 1:
                anomaly_type = self._get_anomaly_type_from_path(pcd_path)
                if anomaly_type not in self.know_class:
                    label = 2

            feature_path = self._get_feature_cache_path(pcd_path, label)

            if not os.path.exists(feature_path):
                all_cache_exists = False
                break

        if all_cache_exists:
            print(f"✔ All {self.xyz_backbone} feature caches exist, skipping model loading")
            return

        self.feature_extractor = PatchCore(self.device)
        self.feature_extractor.load(
            backbone=None,
            layers_to_extract_from=[],
            device=self.device,
            input_shape=(1, 3, 224, 224),
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            basic_template=self.basic_template,
            xyz_backbone_name=self.xyz_backbone
        )
        self.feature_extractor.set_deep_feature_extractor()
        print(f"✔ Feature extractor initialized: {self.xyz_backbone}")

        AnomalyShapeNet._shared_feature_extractor = self.feature_extractor
        AnomalyShapeNet._shared_backbone_name = self.xyz_backbone
        AnomalyShapeNet._shared_device = self.device
 

if __name__ == "__main__":
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
    sys.path.append(PROJECT_ROOT)

    from torch.utils.data import DataLoader
    from types import SimpleNamespace

    args = SimpleNamespace(
        classname="chair0",
        know_class=["bulge"],
        nAnomaly=2,
        dataset_root="/home/zlc/Dataset/Anomaly-ShapeNet-v2/dataset/new_pcd",
        ramdn_seed=42,
        device="cuda:0",
        xyz_backbone='Point_BERT'
    )

    print("===== Testing training set =====")
    train_dataset = AnomalyShapeNet(args, train=True)
    print(f"Training set samples: {len(train_dataset)}")

    print("\n===== Testing test set =====")
    test_dataset = AnomalyShapeNet(args, train=False)
    print(f"Test set samples: {len(test_dataset)}")

    print("\n===== Testing DataLoader =====")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}: pcd shape={batch['pcd'].shape}, labels={batch['label']}")
        if i >= 2:
            break
