import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd

from mnist.model.cnn_model import model, device

# Load the saved state dictionary (make sure 'model_weights.pth' is in your working directory)
model.load_state_dict(torch.load('model_weights.pth'))

# Set the model to evaluation mode
model.eval()

def inference_from_image(image_data: any, invert: bool = False, full_output: bool = False):
  # # Do something interesting with the image data and paths
  if image_data is not None:

    # If the image includes an alpha channel (RGBA), remove it by taking only the first three channels.
    if image_data.shape[-1] == 4:
        image_data = image_data[:, :, :3]

    image = Image.fromarray(image_data.astype('uint8'), 'RGB').convert("L").resize((28, 28))
    image = ImageOps.invert(image) if invert else image

    # Define the same transformation pipeline used during training
    transform = transforms.Compose([
      transforms.ToTensor(),        # Convert image to tensor with shape [C, H, W]
      transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
    ])

    # Apply the transformations
    input_tensor = transform(image)

    # # Add a batch dimension (model expects [N, C, H, W])
    input_tensor = input_tensor.unsqueeze(0)

    # change to the selected input device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
      output = model(input_tensor)
      probabilities = F.softmax(output, dim=1)

      if full_output: 
        probabilities = probabilities.squeeze().tolist() 
        return probabilities
        # for digit, prob in enumerate(probabilities):
        #   print(f"Digit {digit}: {prob * 100:.2f}%")
      else:
        confidence, prediction = torch.max(probabilities, dim=1)
        return (prediction.item(), confidence.item())
  return nil
