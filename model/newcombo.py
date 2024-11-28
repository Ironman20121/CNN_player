import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, input_channels=3):
        super(SpatialTransformer, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # Adjust based on input size
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Compute the transformation parameters
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # Apply the transformation to the input feature map
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.stn = SpatialTransformer()  # Add the STN module
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.1)
        )
        # self.fc1 = nn.Linear(256,256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply Spatial Transformer Network
        x = self.stn(x)
        # Extract features with CNN
        x = self.features(x)
        x = torch.flatten(x, 1)
        # Fully connected layer for classification
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc(x)
        return x
