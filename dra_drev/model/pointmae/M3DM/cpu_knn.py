import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import numpy as np
import torch


class KNN(nn.Module):

    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                point_cloud = ref[bi]
                sample_points = query[bi]
                point_cloud = point_cloud.detach().cpu()
                sample_points = sample_points.detach().cpu()
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(point_cloud.float(), point_cloud.float())
                distances, indices = knn.kneighbors(sample_points, n_neighbors=self.k)

                D.append(distances)
                I.append(indices)
            D = torch.from_numpy(np.array(D))
            I = torch.from_numpy(np.array(I))
        return D, I


def fill_missing_values(x_data, x_label, y_data, k=1):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)

    distances, indices = nn.kneighbors(y_data)
    avg_values = np.mean(x_label[indices], axis=1)
    return avg_values
