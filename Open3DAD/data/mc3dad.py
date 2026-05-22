import os
import glob
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATASETS_PATH = "/home/zlc/Dataset/open-industry/V2"

import os, glob, json, random, hashlib

def stable_hash_int(s: str) -> int:
    # Stable hash: do not use python built-in hash()
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def get_manifest_path(root: str, class_name: str, seed: int, k: int) -> str:
    # Change to relative project directory ./logs
    meta_dir = os.path.join("logs", "mc3dad",str(seed))
    os.makedirs(meta_dir, exist_ok=True)
    return os.path.join(meta_dir, f"pollution_seed{seed}_k{k}.json")

def save_manifest(path: str, root: str, selected_paths: list[str], seed: int, k: int, known_defects: list[str]):

    # Store relative paths more reliably
    rel = [os.path.relpath(p, root) for p in selected_paths]

    obj = {
        "seed": int(seed),
        "k": int(k),
        "known_defects": list(known_defects),
        "selected_relpaths": rel,
    }

    # If file exists and is not empty, read it first
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list")
    else:
        data = []

    # Append
    data.append(obj)

    # Overwrite write back (ensure file is always valid JSON)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_manifest(path: str, seed: int | None = None, k: int | None = None, known_defects: list[str] | None = None):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        return set()

    with open(path, "r") as f:
        obj = json.load(f)

    # Old format: single dict
    if isinstance(obj, dict):
        return set(obj.get("selected_relpaths", []))

    # New format: list[dict]
    if isinstance(obj, list):
        # If no filter conditions, merge and return all records
        if seed is None and k is None and known_defects is None:
            out = set()
            for rec in obj:
                if isinstance(rec, dict):
                    out.update(rec.get("selected_relpaths", []))
            return out

        kd = None if known_defects is None else sorted(list(known_defects))

        # Priority match seed/k/known_defects
        for rec in reversed(obj):  # reversed: most recent write priority
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

def select_pollution_txts(root: str, class_name: str, defect: str, k: int, seed: int) -> list[str]:
    pattern = os.path.join(root, class_name, "gt", f"{class_name}_{defect}*.txt")
    txts = sorted(glob.glob(pattern))

    if k <= 0:
        return []
    assert k <= len(txts), f"[{class_name}] defect={defect}: k={k} > available={len(txts)}"

    # Each defect has independent rng: avoid inconsistency due to call order
    rng = random.Random(int(seed) + stable_hash_int(defect))
    return rng.sample(txts, k=k)


def mbc3dad_classes():
    return [
        "fangxiedianpian",
        "kaikouxiao",
        "koujinluomu",
        "luowending",
        "Lxingluosi",
        "meihuadangquan",
        "neichidianquan",
        "shuangerdianquan",
        "sifangdutu",
        "Sxinggou",
        "waichidianquan",
        "wailiujiaodutou",
        "yangyanzigong",
        "yualuomu",
        "zhongxingdianquan",
    ]


def mbc3dad_defect_types():
    return [
        "good",
        "Bump",
        "Deformation",
        "Dent",
        "Scar",
        "Scratch",
    ]


class MBC3DADBase(Dataset):
    """
    Base dataset:
    - Holds common parsing / IO for .pcd and .txt
    - Provides unified __len__ and __getitem__
    Subclasses only need to build:
      self.sample_paths: List[str]
      self.sample_labels: List[int]  (0=good, 1=anomaly)
    """

    def __init__(self, split: str, class_name: str, root: str = DATASETS_PATH):
        self.split = split
        self.class_name = class_name
        self.root = root
        self.data_path = os.path.join(root, class_name, split)

        # to be filled by subclasses
        self.sample_paths = []
        self.sample_labels = []

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        path = self.sample_paths[idx]
        sample_label = self.sample_labels[idx]  # 0 good / 1 anomaly

        points, mask = self._load_points_and_mask(path)
        points = self._center(points)

        return points, mask, sample_label, path

    # ---------------- IO helpers ----------------
    @staticmethod
    def _center(points: np.ndarray) -> np.ndarray:
        # points: (N,3)
        return points - np.mean(points, axis=0, keepdims=True)

    def _load_points_and_mask(self, path: str):
        """
        Returns:
          points: (N,3) float32
          mask:   (1,N) float32  1=anomaly point, 0=normal point
        Rules:
          - .pcd: no point-level annotation -> mask all zeros
          - .txt: each row: x y z label, label=1 for anomaly, nan for normal
        """
        ext = os.path.splitext(path)[-1].lower()

        if ext == ".pcd":
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points, dtype=np.float32)
            mask = np.zeros((1, points.shape[0]), dtype=np.float32)
            return points, mask

        if ext == ".txt":
            return self._load_txt_xyz_with_nan_label(path)

        raise ValueError(f"Unsupported file format: {path}")

    @staticmethod
    def _load_txt_xyz_with_nan_label(path: str):
        """
        txt: each row: x y z label
          label: 1 -> anomaly point
                 nan/0 -> normal point
        Returns:
          points: (N,3) float32
          mask:   (1,N) float32
        """
        data = np.genfromtxt(path, dtype=np.float32)
        data = np.atleast_2d(data)

        if data.shape[1] < 4:
            raise ValueError(f"{path} needs 4 columns (x y z label), got {data.shape[1]}")

        points = data[:, :3].astype(np.float32)
        lab = data[:, 3].astype(np.float32)

        # Robust: finite & >0.5 => anomaly, else normal (nan/inf -> normal)
        mask_1d = (np.isfinite(lab) & (lab > 0.5)).astype(np.float32)  # (N,)
        return points, mask_1d[None, :]  # (1,N)


class MBC3DADTrain(MBC3DADBase):
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

        # 1) good
        train_dir = os.path.join(self.root, name, "train")
        good_pcds = sorted(glob.glob(os.path.join(train_dir, "*.pcd")))
        paths += good_pcds
        labels += [0] * len(good_pcds)
        print(f"[Train] {name}: good -> {len(good_pcds)} samples")

        # 2) random pollution (per known defect)
        selected = []
        for defect in self.known_defects:
            chosen = select_pollution_txts(
                root=self.root, class_name=name, defect=defect,
                k=self.pollution_per_defect, seed=self.seed
            )
            paths += chosen
            labels += [1] * len(chosen)
            selected += chosen
            if chosen:
                print(f"[Train] {name}: {defect} -> {len(chosen)} samples (random)")

        # 3) save manifest
        manifest_path = get_manifest_path(self.root, name, self.seed, self.pollution_per_defect)
        save_manifest(manifest_path, self.root, selected, self.seed, self.pollution_per_defect, self.known_defects)
        print(f"[Train] {name}: saved manifest -> {manifest_path} (selected={len(selected)})")

        return paths, labels, selected



class MBC3DADTest(MBC3DADBase):
    def __init__(self, class_name: str, known_defects=None, pollution_per_defect: int = 0,
                 root: str = DATASETS_PATH, seed: int = 0 ):
        super().__init__(split="test", class_name=class_name, root=root)
        self.known_defects = known_defects or []
        self.pollution_per_defect = int(pollution_per_defect)
        self.seed = int(seed)

        self.sample_paths, self.sample_labels = self._build_index()
    def _build_index(self):
        paths, labels = [], []
        name = self.class_name

        # 1) good
        test_dir = os.path.join(self.root, name, "test")
        good_pcds = sorted(glob.glob(os.path.join(test_dir, f"{name}_[0-9]*.pcd")))
        paths += good_pcds
        labels += [0] * len(good_pcds)
        print(f"[Test] {name}: good -> {len(good_pcds)} samples")

        # 2) load polluted set (relative paths)
        manifest_path = get_manifest_path(self.root, name, self.seed, self.pollution_per_defect)
        polluted_rel = load_manifest(
            manifest_path,
            seed=self.seed,
            k=self.pollution_per_defect,          # called k in your save_manifest
            known_defects=self.known_defects
        )
        print(f"[Test] {name}: loaded manifest -> {manifest_path} (polluted={len(polluted_rel)})")

        removed = 0

        # 3) anomalies
        for defect in mbc3dad_defect_types():
            if defect == "good":
                continue
            txts = sorted(glob.glob(os.path.join(self.root, name, "gt", f"{name}_{defect}*.txt")))

            # remove those used in train
            kept = []
            for p in txts:
                rp = os.path.relpath(p, self.root)
                if rp in polluted_rel:
                    removed += 1
                    continue
                kept.append(p)

            paths += kept
            labels += [1] * len(kept)
            print(f"[Test] {name}: {defect} -> {len(kept)} samples")

        if removed:
            print(f"[Test] {name}: removed train-polluted anomalies -> {removed}")

        return paths, labels



def get_mbc3dad_loader(
    split: str,
    class_name: str,
    known_defects=None,
    pollution_per_defect: int = 0,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 1,
    seed: int = 0
):
    if known_defects is None or pollution_per_defect == 0:
        print("No anomalies added or removed from training/test set")
    if split == "train":
        dataset = MBC3DADTrain(
            class_name=class_name,
            known_defects=known_defects,
            pollution_per_defect=pollution_per_defect,
            seed=seed
        )
    elif split == "test":
        dataset = MBC3DADTest(
            class_name=class_name,
            known_defects=known_defects,
            pollution_per_defect=pollution_per_defect,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    # NOTE: variable-length point clouds => batch_size>1 will need custom collate_fn
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return loader
