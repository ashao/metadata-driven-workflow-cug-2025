import random

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from mdwc2025 import models
from torch.utils.data import DataLoader, random_split


# Train the Model
def train_model(
    train_dataset,
    model_arch,
    val_loader,
    metawriter,
    num_epochs=800,
    lr=1e-3,
    weight_decay=1e-6,
    device_spec=None,
    weights_file=None,
    convergence_criterion=0.4,
    batch_size=65536,
    subset_data = True
):

    train_size = len(train_dataset)
    subset_size = train_size // 10 if subset_data else train_size  # 1/10 of training data
    if device_spec is None:
      device_spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_spec)
    model = getattr(models, model_arch)().to(device)
    if weights_file:
        model.load_state_dict(torch.load(weights_file, weights_only=True))

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )

    # Initial dummy train_loader to estimate steps_per_epoch
    dummy_subset = torch.utils.data.Subset(train_dataset, random.sample(range(train_size), subset_size))
    dummy_loader = DataLoader(dummy_subset, batch_size=batch_size, shuffle=True)

    ## @Rishabh: shouldn't steps_per_epoch  actually be the len(train_loader) below? Fixed
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, steps_per_epoch=len(dummy_loader), epochs=100
    )
    # criterion = nn.SmoothL1Loss(beta=0.1)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        # Randomly sample 1/10 of the training data each epoch
        subset_indices = random.sample(range(train_size), subset_size)
        subset = torch.utils.data.Subset(train_dataset, subset_indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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

        if epoch < 100:
            scheduler.step()

        print(
            f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )
        metawriter.log_metric(
            "training_metrics",
            {
                "train_loss": str(f"{total_loss/len(train_loader):.4f}"),
                "train_epoch": str(epoch),
            },
        )
        metawriter.log_metric(
            "validation_metrics",
            {
                "val_loss": str(f"{val_loss/len(val_loader):.4f}"),
                "train_epoch": str(epoch),
            },
        )

        if val_loss/len(val_loader) < convergence_criterion/len(val_loader):
            break

    metawriter.commit_metrics("training_metrics")
    metawriter.commit_metrics("validation_metrics")
    metawriter.log_execution_metrics("metrics", {
        "final_epoch": str(epoch),
        "final_val_loss": f"{val_loss/len(val_loader):.4f}",
    })

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

    test_loss = test_loss/len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    metawriter.log_execution_metrics(
        "test_metrics", {"Test Loss": str(f"{test_loss / len(test_loader):.4f}")}
    )
    return test_loss
    # metawriter.commit_metrics("test_metrics")

