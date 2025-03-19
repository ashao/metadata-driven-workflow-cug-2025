import random
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from cmflib import cmf
from mdwc2025 import models
from mdwc2025.data.utils import EKE_Dataset
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, random_split


def cmf_init(pipeline_name="EKEResnet_SmallData"):
    metawriter = cmf.Cmf(
        filename="CUG2025_mlmd", pipeline_name=pipeline_name, graph=False
    )

    context = metawriter.create_context(
        pipeline_stage="Training", custom_properties={"name": "CUG_2025"}
    )

    execution = metawriter.create_execution(
        execution_type="Model_Training", custom_properties={"dataset": "SimulatedData"}
    )
    return metawriter


# Train the Model
def train_model(
    train_dataset,
    model_arch,
    val_loader,
    metawriter,
    num_epochs=1000,
    lr=1e-3,
    weight_decay=1e-6,
):

    train_size = len(train_dataset)
    subset_size = train_size // 10  # 1/10 of training data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, model_arch)().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    ## @Rishabh: shouldn't steps_per_epoch  actually be the len(train_loader) below?
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader), epochs=100
    )
    # criterion = nn.SmoothL1Loss(beta=0.1)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        # Randomly sample 1/10 of the training data each epoch
        subset_indices = random.sample(range(train_size), subset_size)
        subset = torch.utils.data.Subset(train_dataset, subset_indices)
        train_loader = DataLoader(subset, batch_size=1024, shuffle=True)

        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            # print(targets.shape)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()

        scheduler.step()

        print(
            f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )

        if val_loss / len(val_loader) < 0.4:
            metawriter.log_metric(
                "training_metrics",
                {
                    "train_loss": str(f"{total_loss/len(train_loader):.4f}"),
                    "train_epoch": str(epoch),
                },
            )

            metawriter.log_metric(
                "Validation_metrics",
                {
                    "val_loss": str(f"{val_loss/len(val_loader):.4f}"),
                    "train_epoch": str(epoch),
                },
            )

            metawriter.log_execution_metrics("metrics", {"Epoch": str(epoch)})
            return model

        metawriter.log_metric(
            "training_metrics",
            {
                "train_loss": str(f"{total_loss/len(train_loader):.4f}"),
                "train_epoch": str(epoch),
            },
        )

        metawriter.log_metric(
            "Validation_metrics",
            {
                "val_loss": str(f"{val_loss/len(val_loader):.4f}"),
                "train_epoch": str(epoch),
            },
        )

    metawriter.log_execution_metrics("metrics", {"Epoch": str(epoch)})
    return model


# Test Function
def test_model(model, test_loader, metawriter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0
    # criterion = nn.SmoothL1Loss(beta=0.1)
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    metawriter.log_execution_metrics(
        "Test_metrics", {"Test Loss": str(f"{test_loss / len(test_loader):.4f}")}
    )


def main(model_arch, datapath, use_full_dataset, test_small=False):

    suffix = "full_dataset" if use_full_dataset else "small_dataset"
    metawriter = cmf_init(pipeline_name=model_name + suffix)
    # Load Data
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
    ds = dataset if use_full_dataset else dataset.truncate()
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
    model = train_model(train_dataset, model_arch, train_loader, val_loader, metawriter)
    metawriter.commit_metrics("training_metrics")
    metawriter.commit_metrics("Validation_metrics")

    model_label = "_".join([model_name, "model", suffix])
    model_path = model_label + ".pth"
    torch.save(model.state_dict(), model_path)

    # Run Test
    test_model(model, test_loader)
    metawriter.log_model(
        path=model_path,
        event="output",
        model_framework="pytorch",
        model_type="Resnet Conv",
        model_name=model_label,
    )


if __name__ == "__main__":
    DATAPATH = "/lustre/data/shao/cug_2024/"
    model_archs = ["EKEResNet", "EKEBottleneckResNet"]
    all_perm = permutations(model_archs, [True, False])

    for arch, use_full_dataset in all_perm:
        main(arch, DATAPATH, use_full_dataset, test_small=False)
