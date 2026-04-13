import os
import glob
import numpy as np
import open3d as o3d


real3d_abnormal_name2id = {
    "good": 0,
    "Bump": 1,
    "Deformation": 2,
    "Dent": 3,
    "Scar": 4,
    "Scratch": 5,
}

real3d_abnormal_id2name = {
    0: "good",
    1: "Bump",
    2: "Deformation",
    3: "Dent",
    4: "Scar",
    5: "Scratch",
}


def parse_label_and_type(sample_path):
    """
    Returns:
        label: 0 normal, 1 abnormal
        anom_type: class id for abnormal, or good id for normal
    """
    filename = os.path.basename(sample_path)
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')

    if len(parts) < 2:
        return 1, 5

    if parts[1].isdigit():
        return 0, real3d_abnormal_name2id['good']
    return 1, real3d_abnormal_name2id[parts[1]]


def build_train_test_lists(
    dataset_dir,
    cls_name,
    selected_anom_types,
    move_ratio=0.4,
    random_seed=42
):
    """
    Build train/test .pcd path lists: all train/*.pcd plus a fraction of
    selected anomaly types from test; remaining test samples stay in test.
    """
    rng = np.random.RandomState(random_seed)

    train_pattern = os.path.join(dataset_dir, cls_name, 'train', '*.pcd')
    base_train_paths = sorted(glob.glob(train_pattern))

    test_pattern = os.path.join(dataset_dir, cls_name, 'test', '*.pcd')
    all_test_paths = sorted(glob.glob(test_pattern))

    test_normal = []
    test_anom_other = []
    test_anom_selected = {}

    for p in all_test_paths:
        label, ab_id = parse_label_and_type(p)
        anom_type = real3d_abnormal_id2name[ab_id]
        if label == 0:
            test_normal.append(p)
        else:
            if anom_type in selected_anom_types:
                test_anom_selected.setdefault(anom_type, []).append(p)
            else:
                test_anom_other.append(p)

    train_from_test = []
    test_anom_kept = []

    print(f"\n[build_train_test_lists] class: {cls_name}")

    for atype, paths in test_anom_selected.items():
        paths = sorted(paths)
        n_total = len(paths)
        n_move = int(n_total * move_ratio)

        idx = np.arange(n_total)
        rng.shuffle(idx)
        move_idx = idx[:n_move]
        keep_idx = idx[n_move:]

        paths = np.array(paths)
        move_paths = paths[move_idx].tolist()
        keep_paths = paths[keep_idx].tolist()

        train_from_test.extend(move_paths)
        test_anom_kept.extend(keep_paths)

        print(
            f"  anomaly type {atype}: total={n_total}, "
            f"moved_to_train={n_move}, kept_in_test={len(keep_paths)}"
        )

    train_paths = base_train_paths + train_from_test
    test_paths = test_normal + test_anom_other + test_anom_kept

    def count_normal_anom(path_list):
        normal_cnt, anom_cnt = 0, 0
        for p in path_list:
            label, _ = parse_label_and_type(p)
            if label == 0:
                normal_cnt += 1
            else:
                anom_cnt += 1
        return normal_cnt, anom_cnt

    train_normal, train_anom = count_normal_anom(train_paths)
    test_normal_cnt, test_anom_cnt = count_normal_anom(test_paths)

    print("\n[split stats]")
    print(f"  train normal: {train_normal}, train abnormal: {train_anom}")
    print(f"  test normal: {test_normal_cnt}, test abnormal: {test_anom_cnt}")
    print(f"  train total: {len(train_paths)}, test total: {len(test_paths)}")

    return train_paths, test_paths


def save_colored_pcd(pointcloud, mask, save_path):
    """Save a colored PLY: blue=normal (mask 0), red=abnormal (mask 1)."""
    mask = mask.reshape(-1)

    colors = np.zeros((pointcloud.shape[0], 3), dtype=np.float32)
    colors[mask == 0] = [0.0, 0.0, 1.0]
    colors[mask == 1] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    o3d.io.write_point_cloud(save_path, pcd)
    print(f"[save_colored_pcd] wrote {save_path}")


if __name__ == "__main__":
    dataset_dir = "/root/autodl-tmp/3D_abnormal_detection/Open-Industry/V2"
    cls_name = "kaikouxiao"
    selected_anom_types = ['Bump', 'Deformation']
    move_ratio = 0.4

    train_paths, test_paths = build_train_test_lists(
        dataset_dir,
        cls_name,
        selected_anom_types=selected_anom_types,
        move_ratio=move_ratio,
        random_seed=42
    )

    print("\nSample train paths (first 10):")
    for p in train_paths[:10]:
        print(" ", p)

    print("\nSample test paths (first 10):")
    for p in test_paths[:10]:
        print(" ", p)

    print("\nDone.")
