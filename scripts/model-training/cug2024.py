import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import torch.optim.lr_scheduler as lr_scheduler

from mdwc2025 import EKE_Dataset
from torch.utils.data import DataLoader, random_split

# Function to compute C dynamically based on the smallest nonzero absolute value in RV_vert_avg
def compute_C(file_path):
    ds = xr.open_dataset(file_path)
    rv_vert_avg = ds['RV_vert_avg'].values.flatten()
    nonzero_values = np.abs(rv_vert_avg[rv_vert_avg != 0])
    C = np.min(nonzero_values) if len(nonzero_values) > 0 else 1.0  # Avoid zero
    return np.log(C + 1)

# ResNet-style Neural Network with Skip Connections
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),  # BatchNorm added
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)  # BatchNorm added
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.fc(x))  # Skip connection

# Bottleneck Residual Block for CNN-based EKEResNet
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, groups=4):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity  # Residual connection
        return self.relu(out)


# Modified EKE_ResNet with Extra Linear Layers
class EKE_ResNet(nn.Module):
    def __init__(self, input_dim=4):
        super(EKE_ResNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.res_block3 = ResidualBlock(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        # **New layers added**
        self.fc3 = nn.Linear(32, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 1)  # Final output layer

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = torch.relu(self.bn2(self.fc2(x)))

        # **New layers**
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x).squeeze()


# CNN-based EKEResNet with Extra Linear Layers
class EKEResNet(nn.Module):
    def __init__(self):
        super(EKEResNet, self).__init__()

        self.transposed_conv1 = nn.ConvTranspose2d(1, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.transposed_conv2 = nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.res_block1 = BottleneckResidualBlock(32, 64, 64)
        self.res_block2 = BottleneckResidualBlock(64, 64, 64)
        self.res_block3 = BottleneckResidualBlock(64, 64, 64)

        self.pool = nn.AdaptiveMaxPool2d((2, 2))

        self.fc1 = nn.Linear(64 * 2 * 2, 32)
        self.bn_fc1 = nn.BatchNorm1d(32)

        # **New layers added**
        self.fc2 = nn.Linear(32, 8)
        self.bn_fc2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)  # Final output layer

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), 1, 1, -1)
        x = self.relu(self.bn1(self.transposed_conv1(x)))
        x = self.relu(self.bn2(self.transposed_conv2(x)))

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc1(self.fc1(x)))

        # **New layers**
        x = self.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)


# Train the Model
def train_model(train_loader, val_loader, num_epochs=1000, lr=1e-3, weight_decay=1e-6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EKEResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=0.9, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader), epochs=100
)
    #criterion = nn.SmoothL1Loss(beta=0.1)
    criterion = nn.MSELoss()

    train_size = len(train_dataset)
    subset_size = train_size // 10  # 1/10 of training data

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
            #print(outputs.shape)
            #print(targets.shape)
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

        print(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    return model

# Test Function
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0
    #criterion = nn.SmoothL1Loss(beta=0.1)
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")


# Load Data
file_path = "/lustre/data/shao/cug_2024/featurized.nc"  # File Path
dataset = EKE_Dataset(file_path)
#dataset = torch.utils.data.Subset(dataset, range(7168)) #Used this only for a quick check on a small dataset.


print(f"Dataset size before filtering: {len(dataset)}")
print(f"Min ln(EKE): {np.min(dataset.target)}, Max ln(EKE): {np.max(dataset.target)}")
print(f"Mean ln(EKE): {np.mean(dataset.target)}, Std ln(EKE): {np.std(dataset.target)}")

# Train/Test Split
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
#print(len(dataset))
train_dataset, val_dataset, test_dataset = random_split(
    dataset.truncate(),
    [train_size, val_size, test_size]
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Train and Save Model
model = train_model(train_loader, val_loader)
torch.save(model.state_dict(), "ekeresnet_model.pth")

# Run Test
test_model(model, test_loader)
