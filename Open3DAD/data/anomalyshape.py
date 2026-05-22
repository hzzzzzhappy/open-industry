import os
import glob
import json
import random
import hashlib
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
import re

# Default to local dataset path, can be overridden by environment variable
DATASETS_PATH = os.environ.get(
    "ANOMALY_SHAPENET_PATH",
    "/home/zlc/Dataset/Anomaly-ShapeNet-v2/dataset/all_pcd",
)


def stable_hash_int(s: str) -> int:
    # Stable hash: do not use python built-in hash()
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def get_manifest_path(root: str, class_name: str, seed: int, k: int) -> str:
    # Store sampling records in project log directory
    meta_dir = os.path.join("logs", "shapenet3d", str(seed))
    os.makedirs(meta_dir, exist_ok=True)
    return os.path.join(meta_dir, f"pollution_seed{seed}_k{k}_{class_name}.json")


def save_manifest(path: str, root: str, selected_paths: list[str], seed: int, k: int, known_defects: list[str]):
    # Store relative paths more reliably
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


def load_manifest(path: str, seed: int | None = None, k: int | None = None, known_defects: list[str] | None = None):
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


def _find_gt_txts(root: str, class_name: str) -> list[str]:
    """Find GT annotation txt (compatible with GT/gt directories)."""
    gt_dir_candidates = [
        os.path.join(root, class_name, "GT"),
    ]
    for gd in gt_dir_candidates:
        if os.path.isdir(gd):
            base_class = re.sub(r"\d+$", "", class_name)
            txts = sorted(glob.glob(os.path.join(gd, f"{base_class}*.txt")))
            if txts:
                return txts

    return []


def select_pollution_txts(root: str, class_name: str, defect: str, k: int, seed: int) -> list[str]:
    txts = [p for p in _find_gt_txts(root, class_name) if f"_{defect}" in os.path.basename(p)]
    txts = sorted(txts)
    # breakpoint()
    if k <= 0:
        return []
    assert k <= len(txts), f"[{class_name}] defect={defect}: k={k} > available={len(txts)}"

    rng = random.Random(int(seed) + stable_hash_int(defect))
    return rng.sample(txts, k=k)


def shapenet3d_classes():
    # Synchronize with actual categories in all_pcd
    return [

        "ashtray0",
        "bag0",
        "bottle0", "bottle1", "bottle3",
        "bowl0", "bowl1", "bowl2", "bowl3", "bowl4", "bowl5",
        "bucket0", "bucket1",
        "cabinet0",
        "cap0", "cap1", "cap2", "cap3", "cap4", "cap5",
        "chair0",
        "cup0", "cup1", "cup2",
        "desk0",
        "eraser0",
        "headset0", "headset1",
        "helmet0", "helmet1", "helmet2", "helmet3",
        "jar0",
        "knife0", "knife1",
        "microphone0", "microphone1",
        "screen0",
        "shelf0",
        "tap0", "tap1",
        "vase0", "vase1", "vase2", "vase3", "vase4", "vase5", "vase6", "vase7", "vase8", "vase9", "vase10",
    ]

def shapenet3d_defect_types():
    return [
        # Defect types counted from GT/gt filenames in dataset (case-sensitive)
        "bending",
        "broken",
        "bulge",
        "concavity",
        "crak",      # Spelled as crak in dataset
        "hole",
        "scratch",   # All lowercase in dataset
    ]


class ShapeNet3DBase(Dataset):
    """
    Unified return:
      points: (N,3) float32
      mask:   (1,N) float32  (anomaly=1, normal=0)
      label:  int   (0 good / 1 anomaly)
      path:   str
    """
    def __init__(self, split: str, class_name: str, root: str = DATASETS_PATH,
                 good_keyword: str = "positive"):
        self.split = split
        self.class_name = class_name
        self.root = root
        self.good_keyword = good_keyword

        self.sample_paths = []
        self.sample_labels = []

    def __len__(self):
        return len(self.sample_paths)

    @staticmethod
    def _center(points: np.ndarray) -> np.ndarray:
        return points - np.mean(points, axis=0, keepdims=True)

    @staticmethod
    def _load_pcd_points(path: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points, dtype=np.float32)

    @staticmethod
    def _load_txt_points_and_mask(path: str):
        """
        Support two types of txt:
          - Comma-separated: x,y,z,label
          - Space-separated: x y z label
        label: >0.5 is considered anomaly
        Return:
          points (N,3), mask (1,N)
        """
        # Try comma-separated first
        data = np.genfromtxt(path, dtype=np.float32, delimiter=",")
        data = np.atleast_2d(data)

        # If not enough columns, fallback to space/any whitespace separator
        if data.shape[1] < 4:
            data = np.genfromtxt(path, dtype=np.float32)  # whitespace
            data = np.atleast_2d(data)

        if data.shape[1] < 4:
            raise ValueError(f"{path} needs >=4 columns (x y z label), got {data.shape[1]}")

        points = data[:, :3].astype(np.float32)
        lab = data[:, 3].astype(np.float32)
        mask_1d = (np.isfinite(lab) & (lab > 0.5)).astype(np.float32)
        return points, mask_1d[None, :]  # (1,N)

    def __getitem__(self, idx):
        path = self.sample_paths[idx]
        label = int(self.sample_labels[idx])

        ext = os.path.splitext(path)[-1].lower()
        if ext == ".pcd":
            points = self._load_pcd_points(path)
            mask = np.zeros((1, points.shape[0]), dtype=np.float32)
        elif ext == ".txt":
            points, mask = self._load_txt_points_and_mask(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        points = self._center(points)
        return points, mask, label, path


class ShapeNet3DTrain(ShapeNet3DBase):
    def __init__(self, class_name: str, known_defects=None, pollution_per_defect: int = 0,
                 root: str = DATASETS_PATH, seed: int = 0):
        super().__init__(split="train", class_name=class_name, root=root)
        self.known_defects = known_defects or []
        self.pollution_per_defect = int(pollution_per_defect)
        self.seed = int(seed)

        self.sample_paths, self.sample_labels, self.polluted_paths = self._build_index()

    def _build_index(self):
        paths, labels = [], []
        name = self.class_name

        # Good samples
        train_dir = os.path.join(self.root, name, "train")
        good_pcds = sorted(glob.glob(os.path.join(train_dir, "*.pcd")))
        paths += good_pcds
        labels += [0] * len(good_pcds)
        print(f"[Train] {name}: good -> {len(good_pcds)} samples")

        # Random pollution: sample several txts from GT by defect type
        selected = []
        for defect in self.known_defects:
            chosen = select_pollution_txts(
                root=self.root, class_name=name, defect=defect,
                k=self.pollution_per_defect, seed=self.seed,
            )
            paths += chosen
            labels += [1] * len(chosen)
            selected += chosen
            if chosen:
                print(f"[Train] {name}: {defect} -> {len(chosen)} samples (random)")

        # Record sampling for test set filtering
        manifest_path = get_manifest_path(self.root, name, self.seed, self.pollution_per_defect)
        save_manifest(manifest_path, self.root, selected, self.seed, self.pollution_per_defect, self.known_defects)
        print(f"[Train] {name}: saved manifest -> {manifest_path} (selected={len(selected)})")

        return paths, labels, selected


class ShapeNet3DTest(ShapeNet3DBase):
    def __init__(self, class_name: str, known_defects=None, pollution_per_defect: int = 0,
                 root: str = DATASETS_PATH, seed: int = 0, good_keyword: str = "positive"):
        super().__init__(split="test", class_name=class_name, root=root, good_keyword=good_keyword)
        self.known_defects = known_defects or []
        self.pollution_per_defect = int(pollution_per_defect)
        self.seed = int(seed)

        self.sample_paths, self.sample_labels = self._build_index()

    def _build_index(self):
        # Directory structure:
        # root/class/test/*.pcd  (good)
        # root/class/GT/*.txt or root/class/gt/*.txt (anomaly)
        test_dir = os.path.join(self.root, self.class_name, "test")
        all_pcds = sorted(glob.glob(os.path.join(test_dir, "*.pcd")))

        # Keep original logic: only take those containing "positive" as good
        good_pcds = [p for p in all_pcds if self.good_keyword in os.path.basename(p)]
        if len(good_pcds) == 0:
            # Fallback: if no positive naming, treat all as good (avoid empty test set)
            good_pcds = all_pcds

        paths = []
        labels = []

        paths += good_pcds
        labels += [0] * len(good_pcds)

        # Read pollution selected in training set, remove corresponding samples from test set
        manifest_path = get_manifest_path(self.root, self.class_name, self.seed, self.pollution_per_defect)
        polluted_rel = load_manifest(
            manifest_path,
            seed=self.seed,
            k=self.pollution_per_defect,
            known_defects=self.known_defects,
        )

        gt_txts = _find_gt_txts(self.root, self.class_name)
        removed = 0
        kept_gt = []
        for p in gt_txts:
            rp = os.path.relpath(p, self.root)
            if rp in polluted_rel:
                removed += 1
                continue
            kept_gt.append(p)

        paths += kept_gt
        labels += [1] * len(kept_gt)

        print(f"[Test] {self.class_name}: good -> {len(good_pcds)} samples")
        print(f"[Test] {self.class_name}: anomaly(txt) -> {len(kept_gt)} samples")
        if removed:
            print(f"[Test] {self.class_name}: removed train-polluted anomalies -> {removed}")

        assert len(paths) == len(labels)
        return paths, labels


def get_shapenet_loader(
    split: str,
    class_name: str,
    root: str = DATASETS_PATH,
    known_defects=None,
    pollution_per_defect: int = 0,
    seed: int = 0,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 1,
    good_keyword: str = "positive",
):
    if known_defects is None or pollution_per_defect == 0:
        print("No anomalies added or removed from training/test set")

    if split == "train":
        dataset = ShapeNet3DTrain(
            class_name=class_name,
            known_defects=known_defects,
            pollution_per_defect=pollution_per_defect,
            root=root,
            seed=seed,
        )
    elif split == "test":
        dataset = ShapeNet3DTest(
            class_name=class_name,
            known_defects=known_defects,
            pollution_per_defect=pollution_per_defect,
            root=root,
            seed=seed,
            good_keyword=good_keyword,
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return loader
