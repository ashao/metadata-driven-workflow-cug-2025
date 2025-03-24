import torch
import torch.nn as nn

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
class EKEResNet(nn.Module):
    def __init__(self, input_dim=4):
        super(EKEResNet, self).__init__()
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
class EKEBottleneckResNet(nn.Module):
    def __init__(self):
        super(EKEBottleneckResNet, self).__init__()

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
