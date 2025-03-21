import pickle
import pathlib

import numpy as np

from mdwc2025.data.utils import EKE_Dataset

DATAPATH = pathlib.Path("/lustre/data/shao/cug_2024/")
SIMULATION_DATA = DATAPATH / "featurized.nc"
TRAINING_DATA = DATAPATH / "training_data.pkl"
OUTPUT_CLUSTER = DATAPATH / "clusters.pkl"
ds = EKE_Dataset(SIMULATION_DATA)

ds.truncate()
with open(TRAINING_DATA, "wb") as f:
  pickle.dump(ds, f)

labels = ds.clusters.predict(ds.features)
print(f"Samples in excluded cluster: {np.sum(labels == ds.excluded_cluster)}")