import numpy as np
import torch
import xarray as xr

from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from .transforms import symmetric_log, inverse_symmetric_log

class MappableDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.target[idx], dtype=torch.float32)

class EKE_Dataset(MappableDataset):
    def __init__(self, file_path, transform=True):
        self.ds = xr.open_dataset(file_path)
        self.C = self.compute_C() # TODO: Check to see if we should be dynamically changing this

        # Extract features & target, flattening them to 1D vectors
        features = np.stack([
            # using log1p to handle extermely small values that are close to 0
            np.log1p(self.ds['KE_vert_sum'].values.flatten()),  # Log transform
            symmetric_log(self.ds['RV_vert_avg'].values.flatten(), self.C),  # Symmetric Log
            np.log1p(self.ds['slope_vert_avg'].values.flatten()),  # Log transform
            self.ds['Rd_dx_scaled'].values.flatten()  # No log, just normalize later
        ], axis=1)

        target = np.log1p(self.ds['EKE'].values.flatten())  # Log transform target

        # **Filter out samples where ln(EKE) < 0**
        valid_indices = target > 0
        super().__init__(features[valid_indices], target[valid_indices])

        # Compute mean & std for standardization (across dataset)
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)

        self.transform = transform
        if transform:
            self.features = (self.features - self.mean) / self.std

    def __len__(self):
        return len(self.target)

    # Undo the transofrm
    def inverse_transform(self, X):
        if self.transform:
            Y = X*self.std + self.mean
            Y[:,0] = np.expm1(Y[:,0])
            Y[:,1] = inverse_symmetric_log(Y[:,1], self.C)
            Y[:,2] = np.expm1(Y[:,2])
            return Y
        return X

    # Function to compute C dynamically based on the smallest nonzero absolute value in RV_vert_avg
    def compute_C(self):
        rv_vert_avg = self.ds['RV_vert_avg'].values.flatten()
        nonzero_values = np.abs(rv_vert_avg[rv_vert_avg != 0])
        C = np.min(nonzero_values) if len(nonzero_values) > 0 else 1.0  # Avoid zero
        return np.log(C + 1)

    # Return a truncated dataset by deliberating excluding a cluster of data
    # Default is the most positive relative vorticity
    def truncate(self, feature_idx=1):
        clusters = KMeans(n_clusters=6, random_state=0).fit(self.features)
        centers_dimensional = self.inverse_transform(clusters.cluster_centers_)
        excluded_cluster = np.argmax(centers_dimensional[:,feature_idx])
        retained_idx = clusters.labels_ != excluded_cluster
        return MappableDataset(self.features[retained_idx], self.target[retained_idx])

