import torch
import torch.nn as nn
import torch.nn.functional as F


# Redefining the CNN architecture using Conv2d for 2D inputs of size 22x1000 (channels x time points)
class deepCNN(nn.Module):
    def __init__(self):
        super(deepCNN, self).__init__()

        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=32, momentum=0.1),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
            # nn.ELU(alpha=0.9, inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=0.5)
        )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout2d(p=0.5)
        )

        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=128),

            nn.Dropout2d(p=0.5)
        )

        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=256),

            nn.Dropout2d(p=0.5)
        )
        # Linear classification layer
        # The number of linear units will be determined after we know the size of the flattened feature map
        self.fc1 = None  # Will be initialized after the first forward pass
        self.fc2 = nn.Linear(16, 4)

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
            self.fc1 = nn.Linear(x.size(1), 16)  # Number of features by number of classes
            self.fc1 = self.fc1.to(x.device)  # Move to the same device as x

        # Classification layer
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
