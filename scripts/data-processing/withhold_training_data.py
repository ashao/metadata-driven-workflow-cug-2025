import pickle
import pathlib

import numpy as np

from mdwc2025.data.utils import EKE_Dataset

DATAPATH = pathlib.Path("/lustre/data/shao/cug_2024/model_data")
SIMULATION_DATA = DATAPATH / "featurized.nc"
ds = EKE_Dataset(SIMULATION_DATA)

truncated_ds, excluded_ds = ds.truncate()
with open(DATAPATH / "training_data.pkl", "wb") as f:
  pickle.dump(ds, f)

with open(DATAPATH / "truncated_data.pkl", "wb") as f:
  pickle.dump(truncated_ds, f)

with open(DATAPATH / "excluded_data.pkl", "wb") as f:
  pickle.dump(excluded_ds, f)

num_full = np.sum(ds.clusters.predict(ds.features) == ds.excluded_cluster)
print(f"Samples in excluded cluster from full dataset: {num_full}/{len(ds)}")

num_truncated = np.sum(truncated_ds.clusters.predict(truncated_ds.features) == ds.excluded_cluster)
print(f"Samples in excluded cluster from truncated dataset: {num_truncated}/{len(truncated_ds)}")
assert num_truncated == 0

num_excluded = np.sum(excluded_ds.clusters.predict(excluded_ds.features) == ds.excluded_cluster)
print(f"Samples in excluded cluster from excluded dataset: {num_excluded}/{len(excluded_ds)}")
assert num_excluded == num_full
assert num_full > 0