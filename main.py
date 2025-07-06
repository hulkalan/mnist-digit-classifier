import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained MNIST model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Streamlit UI
st.title("✍️ Hand-Drawn Digit Classifier (MNIST)")
st.write("Upload a digit image (preferably drawn in Paint with **white background** and **black digit**)")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image and convert to grayscale
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Original Uploaded Image", use_container_width=True)

    # Resize to 28x28
    img_resized = img.resize((28, 28))

    # Invert image (white → black background, black → white digit)
    img_inverted = ImageOps.invert(img_resized)

    # Display processed image
    st.image(img_inverted, caption="Processed Image for Model", width=150)

    # Normalize and reshape
    img_array = np.array(img_inverted) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = prediction[0][predicted_digit]

    # Output
    st.markdown(f"### ✅ Predicted Digit: **{predicted_digit}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
