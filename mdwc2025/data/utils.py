import copy
import numpy as np
import torch
import xarray as xr

from collections.abc import Iterable

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

    def copy_and_modify(self, features, target):
        new_ds = copy.copy(self)
        new_ds.features = features
        new_ds.target = target
        return new_ds

    def append(self, features, target):
        self.features = np.vstack([self.features, features])
        self.target = np.append(self.target, target)

class EKE_Dataset(MappableDataset):
    mean = None
    std = None
    C = None
    vars = None
    def __init__(self, file_path, do_normalization=True):
        self.do_normalization = do_normalization

        ds = xr.open_dataset(file_path)
        # TODO: Check to see if we should be dynamically changing this
        self.C = self._compute_C(ds.RV_vert_avg.values.flatten())

        self.vars = ["KE_vert_sum", "RV_vert_avg", "slope_vert_avg", "Rd_dx_scaled"]
        # Extract features & target, flattening them to 1D vectors
        features = np.stack(
            [ds[var].values.flatten() for var in self.vars],
            axis=1
        )
        # Compute mean & std for standardization (across dataset)
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)

        features = self.transform(features)
        target = np.log1p(ds['EKE'].values.flatten())  # Log transform target

        # **Filter out samples where ln(EKE) < 0**
        valid_indices = target > 0
        super().__init__(features[valid_indices,:], target[valid_indices])

    def __len__(self):
        return len(self.target)

    def reset_from_dataset(self, ds):
        features = np.stack(
            [ds[var].values.flatten() for var in self.vars],
            axis=1
        )
        target = np.log1p(ds['EKE'].values.flatten())  # Log transform target
        valid_indices = target > 0
        self.features = self.transform(features[valid_indices,:])
        self.target = target[valid_indices]

    def normalize(self, X):
        return (X - self.mean)/self.std

    def inverse_normalize(self, X):
        return X*self.std + self.mean

    def transform(self, X):
        Y = np.zeros_like(X)
        Y[:,0] = np.log1p(X[:,0])
        Y[:,1] = symmetric_log(X[:,1], self.C)  # Symmetric Log
        Y[:,2] = np.log1p(X[:,2])
        Y[:,3] = X[:,3]

        if self.do_normalization:
            Y = self.normalize(Y)
        return Y

    # Undo the transform
    def inverse_transform(self, X):
        if self.do_normalization:
            Y = self.inverse_normalize(X)
        else:
            Y = X.copy()

        Y[:,0] = np.expm1(Y[:,0])
        Y[:,1] = inverse_symmetric_log(Y[:,1], self.C)
        Y[:,2] = np.expm1(Y[:,2])
        return Y

    # Function to compute C dynamically based on the smallest nonzero absolute value in RV_vert_avg
    def _compute_C(self, RV):
        nonzero_values = np.abs(RV[RV != 0])
        C = np.min(nonzero_values) if len(nonzero_values) > 0 else 1.0  # Avoid zero
        return np.log(C + 1)

    # Return a truncated dataset by deliberating excluding a cluster of data
    # Default is the most positive relative vorticity
    def truncate(self, feature_idx=1):
        clusters = KMeans(n_clusters=6, random_state=0).fit(self.features)
        centers_dimensional = self.inverse_transform(clusters.cluster_centers_)
        excluded_cluster = np.argmax(centers_dimensional[:,feature_idx])
        self.clusters = clusters
        self.excluded_cluster = excluded_cluster

        excluded_idx = clusters.labels_ == excluded_cluster
        excluded_ds = self.copy_and_modify(
            self.features[excluded_idx,:], self.target[excluded_idx]
        )

        truncated_idx = clusters.labels_ != excluded_cluster
        truncated_ds = self.copy_and_modify(
            self.features[truncated_idx,:], self.target[truncated_idx]
        )
        return truncated_ds, excluded_ds
