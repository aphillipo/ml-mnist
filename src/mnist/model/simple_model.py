import torch
import torch.nn as nn

# Define the same model architecture used during training
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()             # Flatten 28x28 images to a 784 vector
        self.fc1 = nn.Linear(28 * 28, 128)        # First fully connected layer
        self.relu = nn.ReLU()                    # Activation function
        self.fc2 = nn.Linear(128, 10)             # Output layer: 10 classes for digits 0-9

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = SimpleNet().to(device)
