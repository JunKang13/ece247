import torch
import torch.nn as nn
import torch.nn.functional as F


# Redefining the CNN architecture using Conv2d for 2D inputs of size 22x1000 (channels x time points)
class deepCNN(nn.Module):
    def __init__(self):
        super(deepCNN, self).__init__()

        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, momentum=0.15),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(22, 1)),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, momentum=0.15),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout2d(p=0.4)
        )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, momentum=0.15),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout2d(p=0.4)
        )

        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, momentum=0.15),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout2d(p=0.4)
        )

        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=200, momentum=0.15),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout2d(p=0.4)
        )
        # Linear classification layer
        # The number of linear units will be determined after we know the size of the flattened feature map
        self.fc1 = None  # Will be initialized after the first forward pass
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        # First convolution and max pooling layers
        x = self.ConvLayer1(x)
        # Subsequent convolution and max pooling layers
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)

        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)

        # Initialize the fc layer if it's the first forward pass
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 100)  # Number of features by number of classes
            self.fc1 = self.fc1.to(x.device)  # Move to the same device as x

        # Classification layer
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
