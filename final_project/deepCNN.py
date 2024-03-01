import torch
import torch.nn as nn
import torch.nn.functional as F

# Redefining the CNN architecture using Conv2d for 2D inputs of size 22x1000 (channels x time points)
class deepCNN(nn.Module):
    def __init__(self):
        super(deepCNN, self).__init__()
        
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(22, 1)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        )

        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 10))
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 10))
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 10))
        
        # Linear classification layer
        # The number of linear units will be determined after we know the size of the flattened feature map
        self.fc = None  # Will be initialized after the first forward pass

    def forward(self, x):        
        # First convolution and max pooling layers
        x = self.ConvLayer1(x)
        
        # Subsequent convolution and max pooling layers
        x = self.elu(self.conv2(x))
        x = self.maxpool(x)
        x = self.elu(self.conv3(x))
        x = self.maxpool(x)
        x = self.elu(self.conv4(x))
        x = self.maxpool(x)
        
        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)
        
        # Initialize the fc layer if it's the first forward pass
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), 4)  # Number of features by number of classes
            self.fc = self.fc.to(x.device)  # Move to the same device as x
        
        # Classification layer
        x = self.fc(x)
        return F.softmax(x, dim=1)

