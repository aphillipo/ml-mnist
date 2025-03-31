import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from mnist.model.cnn_model import model, device
from mnist.model.inference import inference_from_image

from mnist.ui.database import save_history

# Load the saved state dictionary (make sure 'model_weights.pth' is in your working directory)
# hack here for if we are inside our docker container
# also we will add different devices, mps for macs, cpu and potentially cuda
inside_docker = os.environ.get('INSIDE_DOCKER', 0) == 1
model_weights_path = f"/app/model_weights.{device}.pth" if inside_docker else f'model_weights.{device}.pth'
model.load_state_dict(torch.load(model_weights_path))

# Set the model to evaluation mode
model.eval()

st.title("Digit Recognizer")

if "data" not in st.session_state:
  st.session_state.prediction = 0
  st.session_state.confidence = 0
  st.session_state.true_value = 0

col1, col2 = st.columns(2)

with col1:
  # Create a canvas component
  canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFF",
    background_color="#000",
    update_streamlit=True,
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
  )

  # # Do something interesting with the image data and paths
  if canvas_result is not None and canvas_result.image_data is not None:
    # Ensure the data is in the proper uint8 format
    img_array = canvas_result.image_data
    prediction, confidence = inference_from_image(canvas_result.image_data)

    st.session_state.prediction = prediction
    st.session_state.confidence = confidence


with col2:
  st.text(f"Predction: {st.session_state.prediction}")
  st.text(f"Confidence: {st.session_state.confidence * 100:.2f}%")

  true_value = st.number_input("True value:", 0, 9, step=1)

  st.button("Submit", None, None, save_history)