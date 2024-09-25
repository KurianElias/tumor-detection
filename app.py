import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\kuria\OneDrive\Desktop\DL\deep-learning\CNN\tumor_detection\myresults\model\cnn_tumor.h5')

# Preprocessing function
def preprocess_image(img):
    img = Image.open(img)
    img = img.resize((128, 128))  # Resizing the image to match model input size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.utils.normalize(img, axis=1)  # Normalize the image
    return img

# Prediction function
def make_prediction(img, model):
    res = model.predict(img)
    if res[0][0] > 0.5:  # Adjust this line based on your model output shape
        return "Tumor Detected"
    else:
        return "No Tumor"

# Streamlit app layout
st.title("Brain Tumor Detection")

# Uploading the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess and display the uploaded image
    image = preprocess_image(uploaded_file)
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    if st.button("Predict"):
        result = make_prediction(image, model)
        st.write(result)
