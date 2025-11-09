import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ----------------------------
# Streamlit GUI for Handwritten Digit Recognition
# ----------------------------

st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")

st.title("🧠 Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below and let the CNN model predict it!")

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("handwritten_digit_model.h5")

model = load_cnn_model()

# Canvas for drawing
st.subheader("Draw your digit here:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert to image
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = img.resize((28, 28))  # Resize to MNIST format
        img = ImageOps.invert(img)  # Invert colors (white digit on black background)
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(img)
        digit = np.argmax(pred)

        st.subheader(f"✨ Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first!")
