import os
import uuid
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
from mnist.ui.database import get_history, init_table, save_history

# Initialize the table only once
if "table_initialized" not in st.session_state:
    init_table()
    st.session_state.table_initialized = True

# Load model weights based on environment
inside_docker = os.environ.get('INSIDE_DOCKER', 0) == 1
model_weights_path = f"/app/model_weights.{device}.pth" if inside_docker else f'model_weights.{device}.pth'
model.load_state_dict(torch.load(model_weights_path))
model.eval()

st.title("Digit Recognizer")

# Initialize session state values only once
if "prediction" not in st.session_state:
    st.session_state.prediction = 0
if "confidence" not in st.session_state:
    st.session_state.confidence = 0
if "true_value" not in st.session_state:
    st.session_state.true_value = 0
if "initial_drawing" not in st.session_state:
    st.session_state.initial_drawing = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = str(uuid.uuid4())
if "true_value_key" not in st.session_state:
    st.session_state.true_value_key = str(uuid.uuid4())

def on_submit():
    # Save current prediction and true value
    save_history(st.session_state.prediction, st.session_state.true_value)
    # Reset state values and generate new keys for the canvas and number input
    st.session_state.prediction = 0
    st.session_state.confidence = 0
    st.session_state.true_value = 0
    st.session_state.initial_drawing = None
    st.session_state.canvas_key = str(uuid.uuid4())
    st.session_state.true_value_key = str(uuid.uuid4())
    st.rerun()

col1, col2 = st.columns(2)

with col1:
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#FFF",
        background_color="#000",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        display_toolbar=False,
        initial_drawing=st.session_state.initial_drawing
    )
    
    # Only update inference if canvas data exists and we're not in a "just submitted" state
    if canvas_result is not None and canvas_result.image_data is not None:
        prediction, confidence = inference_from_image(canvas_result.image_data)
        st.session_state.prediction = prediction
        st.session_state.confidence = confidence

with col2:
    st.text(f"Prediction: {st.session_state.prediction}")
    st.text(f"Confidence: {st.session_state.confidence * 100:.2f}%")
    st.session_state.true_value = st.number_input("True value:", 0, 9, step=1, value=0, key=st.session_state.true_value_key)
    st.button("Submit", on_click=on_submit)

st.text("History")
data = get_history()
df = pd.DataFrame(data, columns=["date", "prediction", "true_value"])
st.dataframe(df)
