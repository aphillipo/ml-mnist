import torch
import torch.nn as nn
import torch.nn.functional as F

def getDevice():
    if (torch.cuda.is_available()):
        return "cuda"
    elif (torch.backends.mps.is_available()):
        return "mps"
    else:
        return "cpu"

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: input channels=1 (grayscale), output channels=32, kernel size 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: input channels=32, output channels=64, kernel size 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer to reduce spatial dimensions by half
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(0.25)
        # Fully connected layers
        # After two poolings, image size goes from 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [batch, 32, 28, 28]
        x = self.pool(x)           # [batch, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [batch, 64, 14, 14]
        x = self.pool(x)           # [batch, 64, 7, 7]
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device(getDevice())

model = CNN().to(device)