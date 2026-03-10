import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load your trained Random Forest model
model = joblib.load("rfc_model.pkl")  # Make sure it's in the same folder

st.title("Car Classifier: Audi vs Toyota")

# Upload image
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess the image to match model input:
    - Resize to target_size
    - Convert to numpy array
    - Flatten to 1D (128*128*3 = 49152 features)
    """
    image = image.convert('RGB')  # Ensure 3 channels
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized)
    return image_array.flatten().reshape(1, -1)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        try:
            features = preprocess_image(image)
            
            prediction = model.predict(features)[0]
            st.success(f"Predicted Class: {prediction}")
        
        except ValueError as e:
            st.error(f"Error: {e}")