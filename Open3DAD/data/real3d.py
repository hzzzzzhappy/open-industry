import glob
import hashlib
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

DATASETS_PATH = '/home/zlc/Dataset/Real3D-AD-PCD'


def stable_hash_int(s: str) -> int:
    # Stable hash: do not use python built-in hash()
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def get_manifest_path(root: str, class_name: str, seed: int, k: int) -> str:
    # Store sampling records in project log directory
    meta_dir = os.path.join("logs", "real3d", str(seed))
    os.makedirs(meta_dir, exist_ok=True)
    return os.path.join(meta_dir, f"pollution_seed{seed}_k{k}_{class_name}.json")


def save_manifest(path: str, root: str, selected_paths: List[str], seed: int, k: int, known_defects: List[str]):
    rel = [os.path.relpath(p, root) for p in selected_paths]

    obj = {
        "seed": int(seed),
        "k": int(k),
        "known_defects": list(known_defects),
        "selected_relpaths": rel,
    }

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list")
    else:
        data = []

    data.append(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_manifest(path: str, seed: int | None = None, k: int | None = None, known_defects: List[str] | None = None):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        return set()

    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        return set(obj.get("selected_relpaths", []))

    if isinstance(obj, list):
        if seed is None and k is None and known_defects is None:
            out = set()
            for rec in obj:
                if isinstance(rec, dict):
                    out.update(rec.get("selected_relpaths", []))
            return out

        kd = None if known_defects is None else sorted(list(known_defects))

        for rec in reversed(obj):
            if not isinstance(rec, dict):
                continue

            if seed is not None and int(rec.get("seed", -1)) != int(seed):
                continue
            if k is not None and int(rec.get("k", -1)) != int(k):
                continue
            if kd is not None and sorted(rec.get("known_defects", [])) != kd:
                continue

            return set(rec.get("selected_relpaths", []))

        return set()

    raise ValueError(f"Unsupported manifest format in {path}: {type(obj)}")


def _find_gt_txts(root: str, class_name: str, defect: str) -> List[str]:
    defect_dir = os.path.join(root, class_name, "gt")
    if not os.path.isdir(defect_dir):
        return []

    txts = sorted(glob.glob(os.path.join(defect_dir, "*.txt")))
    if not defect:
        return txts

    matched = []
    for p in txts:
        stem = os.path.splitext(os.path.basename(p))[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        core = parts[1:]
        if core and core[-1].lower() == "cut":
            core = core[:-1]
        if not core:
            continue
        defect_name = "_".join(core)
        if defect_name == defect:
            matched.append(p)
    return matched


def select_pollution_txts(root: str, class_name: str, defect: str, k: int, seed: int) -> List[str]:
    txts = _find_gt_txts(root, class_name, defect)
    # breakpoint()
    if k <= 0:
        return []
    assert k <= len(txts), f"[{class_name}] defect={defect}: k={k} > available={len(txts)}"

    rng = random.Random(int(seed) + stable_hash_int(defect))
    return rng.sample(txts, k=k)


def real3d_classes():
    return [
        "airplane",
        "candybar",
        "car",
        "chicken",
        "diamond",
        "duck",
        "fish",
        "gemstone",
        "seahorse",
        "shell",
        "starfish",
        "toffees",
    ]


def real3d_defect_types(root: str = DATASETS_PATH, classes: List[str] | None = None) -> List[str]:
    """Scan gt directory and count all defect types (extracted from filenames).

    Convention: filenames are like `123_type.txt` or `123_type_cut.txt`,
    only take the part after the number, remove trailing `_cut`.
    """
    cls_list = classes if classes is not None else real3d_classes()
    defects = set()

    for cls in cls_list:
        gt_dir = os.path.join(root, cls, "gt")
        if not os.path.isdir(gt_dir):
            continue

        for p in glob.glob(os.path.join(gt_dir, "*.txt")):
            base = os.path.splitext(os.path.basename(p))[0]
            parts = base.split("_")
            if len(parts) < 2:
                continue

            # Remove first number, remove trailing cut
            core = parts[1:]
            if core and core[-1].lower() == "cut":
                core = core[:-1]
            if not core:
                continue

            defect = "_".join(core)
            defects.add(defect)

    return sorted(defects)


voxel_size_setting = 0.15


class Real3D(Dataset):

    def __init__(self, split, class_name, root=DATASETS_PATH, known_defects=None, pollution_per_defect=0, seed=0):
        self.cls = class_name
        self.root = root
        self.split = split
        self.data_path = os.path.join(self.root, self.cls, split)
        self.known_defects = known_defects or []
        self.pollution_per_defect = int(pollution_per_defect)
        self.seed = int(seed)

    @staticmethod
    def _load_and_downsample(pcd_path: str, mask_path: Optional[str] = None):
        # Read point cloud
        pcd = o3d.io.read_point_cloud(pcd_path)
        points_all = np.asarray(pcd.points, dtype=np.float32)

        # Downsample
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
        ds_points = np.asarray(pcd_ds.points, dtype=np.float32)

        if ds_points.size == 0:
            return ds_points, torch.zeros((1, 0), dtype=torch.float32)

        if mask_path is None:
            mask = torch.zeros((1, ds_points.shape[0]), dtype=torch.float32)
            return ds_points, mask

        # Read GT, get abnormal point coordinates + labels
        gt_data = np.loadtxt(mask_path, dtype=np.float32)
        gt_data = np.atleast_2d(gt_data)
        if gt_data.shape[1] < 4:
            raise ValueError(f"{mask_path} needs >=4 columns (x y z label)")

        gt_points = gt_data[:, :3].astype(np.float32)
        gt_labels = (gt_data[:, 3] > 0.5).astype(np.float32)

        if gt_points.shape[0] == 0:
            mask = torch.zeros((1, ds_points.shape[0]), dtype=torch.float32)
            return ds_points, mask

        # Nearest neighbor mapping to downsampled points
        gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
        kd = o3d.geometry.KDTreeFlann(gt_pcd)
        mask_list = []
        for x in ds_points:
            _, idx, _ = kd.search_knn_vector_3d(x, 1)
            mask_list.append(gt_labels[idx[0]])

        mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(0)
        return ds_points, mask


class Real3DTrain(Real3D):
    def __init__(self, class_name, known_defects=None, pollution_per_defect=0, root=DATASETS_PATH, seed=0):
        super().__init__(split="train", class_name=class_name, root=root, known_defects=known_defects,
                         pollution_per_defect=pollution_per_defect, seed=seed)
        self.pcd_paths, self.labels, self.polluted_paths = self.load_dataset()

    def load_dataset(self):
        pcd_paths: List[Tuple[str, Optional[str]]] = []  # (pcd, mask)
        labels = []
        selected_gt = []

        train_good_dir = os.path.join(self.root, self.cls, "train")
        good_paths = sorted(glob.glob(os.path.join(train_good_dir, "*.pcd")))
        good_paths += sorted(glob.glob(os.path.join(train_good_dir, "*.asc")))
        for p in good_paths:
            pcd_paths.append((p, None))
            labels.append(0)
        print(f"[Train] {self.cls}: good -> {len(good_paths)} samples")

        for defect in self.known_defects:
            chosen_gt = select_pollution_txts(
                root=self.root,
                class_name=self.cls,
                defect=defect,
                k=self.pollution_per_defect,
                seed=self.seed,
            )

            for gt_path in chosen_gt:
                stem = os.path.splitext(os.path.basename(gt_path))[0]
                pcd_path = os.path.join(self.root, self.cls, "test", f"{stem}.pcd")
                if not os.path.isfile(pcd_path):
                    raise FileNotFoundError(f"Missing pcd for GT {gt_path}: {pcd_path}")

                pcd_paths.append((pcd_path, gt_path))
                labels.append(1)
            selected_gt += chosen_gt

            if chosen_gt:
                print(f"[Train] {self.cls}: {defect} -> {len(chosen_gt)} samples (random)")

        manifest_path = get_manifest_path(self.root, self.cls, self.seed, self.pollution_per_defect)
        save_manifest(manifest_path, self.root, selected_gt, self.seed, self.pollution_per_defect, self.known_defects)
        print(f"[Train] {self.cls}: saved manifest -> {manifest_path} (selected={len(selected_gt)})")

        return pcd_paths, labels, selected_gt

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, mask_path = self.pcd_paths[idx]
        label = self.labels[idx]
        unorganized_pc, mask = self._load_and_downsample(pcd_path, mask_path)
        return unorganized_pc, mask, label, pcd_path


class Real3DTest(Real3D):
    def __init__(self, class_name, known_defects=None, pollution_per_defect=0, root=DATASETS_PATH, seed=0):
        super().__init__(split="test", class_name=class_name, root=root, known_defects=known_defects,
                         pollution_per_defect=pollution_per_defect, seed=seed)
        self.pcd_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        pcd_paths: List[Tuple[str, Optional[str]]] = []
        labels = []

        manifest_path = get_manifest_path(self.root, self.cls, self.seed, self.pollution_per_defect)
        polluted_rel = load_manifest(
            manifest_path,
            seed=self.seed,
            k=self.pollution_per_defect,
            known_defects=self.known_defects,
        )

        test_dir = os.path.join(self.root, self.cls, "test")
        test_pcds = sorted(glob.glob(os.path.join(test_dir, "*.pcd")))

        removed = 0
        for p in test_pcds:
            stem = os.path.splitext(os.path.basename(p))[0]
            is_good = "good" in stem.lower()
            mask_path = None if is_good else os.path.join(self.root, self.cls, "gt", f"{stem}.txt")

            if mask_path and not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Missing GT for {p}: {mask_path}")

            rp = os.path.relpath(mask_path, self.root) if mask_path else None
            if rp and rp in polluted_rel:
                removed += 1
                continue

            pcd_paths.append((p, mask_path))
            labels.append(0 if is_good else 1)

        print(f"[Test] {self.cls}: good -> {sum(1 for _, m in pcd_paths if m is None)} samples")
        print(f"[Test] {self.cls}: anomaly -> {sum(1 for _, m in pcd_paths if m is not None)} samples")
        if removed:
            print(f"[Test] {self.cls}: removed train-polluted anomalies -> {removed}")

        assert len(pcd_paths) == len(labels)
        return pcd_paths, labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, mask_path = self.pcd_paths[idx]
        label = self.labels[idx]
        unorganized_pc, mask = self._load_and_downsample(pcd_path, mask_path)
        return unorganized_pc, mask, label, pcd_path


def get_real_loader(split, class_name, root=DATASETS_PATH, known_defects=None, pollution_per_defect=0, seed=0,
                    batch_size=1, shuffle=False, num_workers=1):
    if known_defects is None or pollution_per_defect == 0:
        print("No anomalies added or removed from training/test set")

    if split == 'train':
        dataset = Real3DTrain(class_name=class_name, known_defects=known_defects, pollution_per_defect=pollution_per_defect,
                              root=root, seed=seed)
    elif split == 'test':
        dataset = Real3DTest(class_name=class_name, known_defects=known_defects, pollution_per_defect=pollution_per_defect,
                     root=root, seed=seed)
    else:
        raise ValueError(f"Unknown split: {split}")

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             drop_last=False, pin_memory=True)
    return data_loader
