import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from cnn_model import model, device

# Load the saved state dictionary (make sure 'model_weights.pth' is in your working directory)
model.load_state_dict(torch.load('model_weights.pth'))

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=7,
    stroke_color="#000",
    background_color="#FFF",
    update_streamlit=True,
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# # Do something interesting with the image data and paths
if canvas_result.image_data is not None:
  # Ensure the data is in the proper uint8 format
  img_array = canvas_result.image_data

  # If the image includes an alpha channel (RGBA), remove it by taking only the first three channels.
  if img_array.shape[-1] == 4:
      img_array = img_array[:, :, :3]

  image = Image.fromarray(img_array.astype('uint8'), 'RGB').convert("L").resize((28, 28))
  image = ImageOps.invert(image) # very important!

  st_image = st.image(image)

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


  # Set the model to evaluation mode
  model.eval()

  with torch.no_grad():
      output = model(input_tensor)
      probabilities = F.softmax(output, dim=1)  # apply softmax along the class dimension
      # predicted_digit = output.argmax(dim=1).item()

  probabilities = probabilities.squeeze().tolist()  # shape: [10] for 10 classes
  for digit, prob in enumerate(probabilities):
    print(f"Digit {digit}: {prob * 100:.2f}%")

