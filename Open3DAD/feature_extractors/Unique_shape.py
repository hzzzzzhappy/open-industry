import numpy as np
from sklearn.neighbors import NearestNeighbors
# from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from feature_extractors.features import *
from feature_extractors.pointnet2_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_normals(points, k=10):
    """Compute normal vectors for each point using PCA on k-nearest neighbors."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    normals = []
    
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        cov_matrix = np.cov(neighbors - np.mean(neighbors, axis=0), rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normal = eigvecs[:, np.argmin(eigvals)]
        normals.append(normal)
    
    return np.array(normals)

def compute_usc_descriptor(points, normals, keypoints, num_bins_r=5, num_bins_theta=12, radius=0.2):
    """Compute Unique Shape Context (USC) descriptor for given keypoints."""
    descriptors = []

    for keypoint in keypoints:
        # Find normal vector closest to keypoint
        idx = np.argmin(np.linalg.norm(points - keypoint, axis=1))
        normal = normals[idx]

        # Compute neighborhood points
        nbrs = NearestNeighbors(radius=100,n_neighbors=40).fit(points)
        indices = nbrs.radius_neighbors([keypoint], return_distance=False)[0]
        local_points = points[indices] - keypoint

        # Compute local coordinate system (normal aligned)
        tangent_x = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(tangent_x) < 1e-6:
            tangent_x = np.cross(normal, [0, 1, 0])
        tangent_x /= np.linalg.norm(tangent_x)
        tangent_y = np.cross(normal, tangent_x)

        # Compute polar coordinates of neighborhood points
        r_vals = np.linalg.norm(local_points, axis=1)
        theta_vals = np.arctan2(np.dot(local_points, tangent_y), np.dot(local_points, tangent_x))

        # Compute histogram indices
        bin_r = np.floor(r_vals / radius * num_bins_r).astype(int)
        bin_theta = np.floor((theta_vals + np.pi) / (2 * np.pi) * num_bins_theta).astype(int)

        # Generate USC histogram
        usc_hist = np.zeros((num_bins_r, num_bins_theta))
        for r, t in zip(bin_r, bin_theta):
            if 0 <= r < num_bins_r and 0 <= t < num_bins_theta:
                usc_hist[r, t] += 1

        descriptors.append(usc_hist.flatten())  # Flatten to vector

    return np.array(descriptors)



def get_USC(unorganized_pc):
    unorganized_pc = unorganized_pc.squeeze(0).numpy()
    print(unorganized_pc.shape)

    normals = compute_normals(unorganized_pc)

    # fps the centers out
    # unorganized_pc_no_zeros expected (1,n,3)
    unorganized_pc_no_zeros = unorganized_pc
    unorganized_pc_no_zeros = torch.tensor(unorganized_pc_no_zeros).cuda().unsqueeze(dim=0)
    unorganized_pc_no_zeros = unorganized_pc_no_zeros.to(torch.float32)

    batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape
    center, center_idx = fps(unorganized_pc_no_zeros.contiguous(), 1024)  # B G 3
    # print(center.shape)
    features = compute_usc_descriptor(unorganized_pc, normals,center.squeeze().cpu().numpy())
    features = torch.tensor(features).cuda()
    agg_point_feature = features
    unorganized_pc = torch.tensor(unorganized_pc)

    # print(agg_point_feature.shape,unorganized_pc.shape,unorganized_pc_no_zeros.shape,center.shape)
    return agg_point_feature,unorganized_pc,unorganized_pc_no_zeros,center





if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(100, 3)  # Generate random point cloud
    normals = compute_normals(points)
    keypoints = points[np.random.choice(len(points), 5, replace=False)]  # Randomly select 5 keypoints
    usc_descriptors = compute_usc_descriptor(points, normals, keypoints)
    print("USC Descriptor Shape:", usc_descriptors.shape)
