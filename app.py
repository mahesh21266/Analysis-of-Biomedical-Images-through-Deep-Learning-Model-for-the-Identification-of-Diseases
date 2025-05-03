import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model (make sure to upload or reference your .h5 model)
model = load_model('bio_medical_model.h5')  # Replace 'your_model.h5' with the actual path to your model

# Display title
st.markdown("<h1 style='text-align: center;'>Bio medical image analsyis Using DL</h1>", unsafe_allow_html=True)

# Add image to the content
image_path = "8-surprising-facts-about-lungs-1692982415-removebg-preview.png"
st.image(image_path, caption='', use_column_width=True)

# File uploader for image input
uploaded_file = st.file_uploader("Please Upload your CT scan here", type=["jpg", "jpeg", "png"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Open the uploaded image using PIL
    img = Image.open(uploaded_file)
    
    # Ensure the image is in RGB format (removes alpha channel if present)
    img = img.convert("RGB")
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image to fit the model's input requirements
    img = img.resize((224, 224))  # Resize image to 224x224 (modify as per your model's input size)
    img_array = np.array(img)  # Convert to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (model expects a batch of images)
    
    # Normalize the image (if necessary, depending on the model)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1] if model expects this

    # Perform model inference (predict)
    predictions = model.predict(img_array)

    # Assuming the model returns a single class output, you can display it
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    class_names = ["adenocarcinoma", "large-cell-carcinoma", "normal", "squamous-cell-carcinoma"]  # Replace these with your actual class names
    st.write(f"Predicted Class: {class_names[predicted_class[0]]}")
