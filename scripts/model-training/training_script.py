import random
from itertools import permutations, product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from cmflib import cmf
from mdwc2025 import models
from mdwc2025.training import test_model, train_model
from mdwc2025.data.utils import EKE_Dataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, random_split


def cmf_init(pipeline_name="EKEResnet_SmallData"):
    metawriter = cmf.Cmf(
        filepath="CUG2025_mlmd", pipeline_name=pipeline_name, graph=False
    )

    context = metawriter.create_context(
        pipeline_stage="Training", custom_properties={"name": "CUG_2025"}
    )

    execution = metawriter.create_execution(
        execution_type="Model_Training", custom_properties={"dataset": "SimulatedData"}
    )
    return metawriter


def main(model_arch, datapath, use_full_dataset, test_small=False):
    suffix = "full_dataset" if use_full_dataset else "small_dataset"
    metawriter = cmf_init(pipeline_name=model_arch + suffix)
    # Load Data

    metawriter.log_dataset(
        datapath,
        "input",
        custom_properties = {"name": str(suffix) + "_simulated"}
        )
    dataset = EKE_Dataset(datapath)
    if test_small:
        dataset = torch.utils.data.Subset(
            dataset, range(7168)
        )  # Used this only for a quick check on a small dataset.

    print(f"Dataset size before filtering: {len(dataset)}")
    print(
        f"Min ln(EKE): {np.min(dataset.target)}, Max ln(EKE): {np.max(dataset.target)}"
    )
    print(
        f"Mean ln(EKE): {np.mean(dataset.target)}, Std ln(EKE): {np.std(dataset.target)}"
    )

    # Train/Test Split
    if use_full_dataset:
        ds = dataset
    else:
        truncated_ds, _ = dataset.truncate()
        ds = truncated_ds

    dataset_size = len(ds)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size  # Ensure exact match

    train_dataset, val_dataset, test_dataset = random_split(
        ds, [train_size, val_size, test_size]
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Train and Save Model
    model = train_model(train_dataset, model_arch, val_loader, metawriter)

    model_label = "_".join([model_arch, "model", suffix])
    model_path = model_label + ".pth"
    # torch.save(model.state_dict(), model_path)

    # Run Test
    test_model(model, test_loader, metawriter=metawriter)
    metawriter.log_model(
        path=model_path,
        event="output",
        model_framework="pytorch",
        model_type="Resnet Conv",
        model_name=model_label,
    )


if __name__ == "__main__":
    DATAPATH = "/lustre/data/shao/cug_2025/featurized.nc"
    model_archs = ["EKEResNet", "EKEBottleneckResNet"]
    all_perm = product(model_archs, [True, False])

    for arch, use_full_dataset in all_perm:
        main(arch, DATAPATH, use_full_dataset, test_small=False)
