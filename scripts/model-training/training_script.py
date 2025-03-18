from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader, random_split
import random
import torch.optim.lr_scheduler as lr_scheduler
from cmflib import cmf


metawriter = cmf.Cmf(
   filename="CUG2024_mlmd",
   pipeline_name="EKEResnet_SmallData",
   graph=False
)     


context = metawriter.create_context(
    pipeline_stage="Training",
    custom_properties={
        "name": "CUG_2024"
    }
)

execution = metawriter.create_execution(
    execution_type="Model_Training",
    custom_properties = {
        "dataset": "SimulatedData"
    }
)

# Function to compute C dynamically based on the smallest nonzero absolute value in RV_vert_avg
def compute_C(file_path):
    ds = xr.open_dataset(file_path)
    rv_vert_avg = ds['RV_vert_avg'].values.flatten()
    nonzero_values = np.abs(rv_vert_avg[rv_vert_avg != 0])
    C = np.min(nonzero_values) if len(nonzero_values) > 0 else 1.0  # Avoid zero
    return np.log(C + 1)

# Symmetric Log Transformation
def symmetric_log(x, C):
    return np.sign(x) * np.log1p(np.abs(x) + C)

# Inverse of the symmetric log
def inverse_symmetric_log(y, C):
    return np.sign(y)*(np.exp(np.sign(y)*y) - C - 1)

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

        if val_loss/len(val_loader) < 0.4:
            metawriter.log_metric("training_metrics", 
                   {"train_loss": str(f"{total_loss/len(train_loader):.4f}"),
                    "train_epoch": str(epoch)}
                ) 
        
            metawriter.log_metric("Validation_metrics", 
                    {"val_loss": str(f"{val_loss/len(val_loader):.4f}"),
                        "train_epoch": str(epoch)}
                    ) 
            
            metawriter.log_execution_metrics(
                "metrics", {"Epoch": str(epoch)}
            )
            return model
        
        metawriter.log_metric("training_metrics", 
                   {"train_loss": str(f"{total_loss/len(train_loader):.4f}"),
                    "train_epoch": str(epoch)}
                ) 
        
        metawriter.log_metric("Validation_metrics", 
                   {"val_loss": str(f"{val_loss/len(val_loader):.4f}"),
                    "train_epoch": str(epoch)}
                ) 
        
    metawriter.log_execution_metrics(
            "metrics", {"Epoch": str(epoch)}
        )
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
    metawriter.log_execution_metrics(
            "Test_metrics", {"Test Loss": str(f"{test_loss / len(test_loader):.4f}")}
        )


# Load Data
file_path = "/lustre/data/shao/cug_2024/featurized.nc"  # File Path
dataset = EKE_Dataset(file_path)
#dataset = torch.utils.data.Subset(dataset, range(7168)) #Used this only for a quick check on a small dataset.


print(f"Dataset size before filtering: {len(dataset)}")
print(f"Min ln(EKE): {np.min(dataset.target)}, Max ln(EKE): {np.max(dataset.target)}")
print(f"Mean ln(EKE): {np.mean(dataset.target)}, Std ln(EKE): {np.std(dataset.target)}")

# Train/Test Split 

# For large dataset uncomment below and comment out the small dataset section
'''
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
#print(len(dataset))

train_dataset, val_dataset, test_dataset = random_split(
    dataset.truncate(),
    [train_size, val_size, test_size]
)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
'''

# For smaller dataset
truncated_dataset = dataset.truncate()
dataset_size = len(truncated_dataset)

# Compute sizes
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size  # Ensure exact match

train_dataset, val_dataset, test_dataset = random_split(
    truncated_dataset,
    [train_size, val_size, test_size]
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Train and Save Model
model = train_model(train_loader, val_loader)
metawriter.commit_metrics("training_metrics")
metawriter.commit_metrics("Validation_metrics")

torch.save(model.state_dict(), "EKEResNet_model_Smalldataset.pth")

# Run Test
test_model(model, test_loader)
metawriter.log_model(
            path="EKEResNet_model_Smalldataset.pth", event="output", model_framework="pytorch", model_type="Resnet Conv", 
            model_name="EKEResNet_model_Smalldataset.pth" 
            )