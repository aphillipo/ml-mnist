import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from cnn_model import model, device

# Path to your image file (ensure it's a 28x28 grayscale image or will be resized)
image_path = './number9.png'

# Open the image and convert to grayscale
image = Image.open(image_path).convert('L')

# Define the same transformation pipeline used during training
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.ToTensor(),        # Convert image to tensor with shape [C, H, W]
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
])

# Apply the transformations
input_tensor = transform(image)

# Add a batch dimension (model expects [N, C, H, W])
input_tensor = input_tensor.unsqueeze(0)

# change to the selected input device
input_tensor = input_tensor.to(device)

# Load the saved state dictionary (make sure 'model_weights.pth' is in your working directory)
model.load_state_dict(torch.load('model_weights.pth'))

# Set the model to evaluation mode
model.eval()

with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)  # apply softmax along the class dimension
    # predicted_digit = output.argmax(dim=1).item()

probabilities = probabilities.squeeze().tolist()  # shape: [10] for 10 classes
for digit, prob in enumerate(probabilities):
  print(f"Digit {digit}: {prob * 100:.2f}%")

