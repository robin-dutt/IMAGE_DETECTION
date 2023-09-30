import streamlit as st
from PIL import Image
import os
import exifread
import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageChops, ImageEnhance
import glob
import random
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Function to perform metadata analysis on an image
def perform_metadata_analysis(image_path):
    # Extract metadata from the image
    def extract_metadata(image_path):
        with open(image_path, 'rb') as image_file:
            tags = exifread.process_file(image_file)
        return tags

    # Extract Creation Date/Time, Software Used, and Date Modified
    def extract_metadata_details(image_path):
        metadata = extract_metadata(image_path)

        creation_time = metadata.get('EXIF DateTimeOriginal', None)
        software_used = metadata.get('Image Software', None)

        # Extract Date Modified using os.path.getmtime and convert to a readable format
        date_modified_timestamp = os.path.getmtime(image_path)
        date_modified = datetime.fromtimestamp(date_modified_timestamp).strftime('%Y-%m-%d %H:%M:%S')

        return {
            'Creation Date/Time': str(creation_time),
            'Software Used': str(software_used),
            'Date Modified': date_modified,
        }

    metadata_details = extract_metadata_details(image_path)
    return metadata_details

# Function to determine whether an image is real or fake
def classify_image(image_path):
    def convert_to_ela_image(image_path, quality=90):
        # Save the image at the given quality
        temp_file = 'temp.jpg'
        im = Image.open(image_path).convert('RGB')
        im.save(temp_file, 'JPEG', quality=quality)

        # Open the saved image and the original image
        saved = Image.open(temp_file)
        original = Image.open(image_path)

        # Find the absolute difference between the images
        diff = ImageChops.difference(original, saved)

        # Normalize the difference by multiplying with a scale factor and convert to grayscale
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)

        # Remove the temporary file
        os.remove(temp_file)

        return diff

    def prepare_image(image_path):
        return np.array(convert_to_ela_image(image_path).resize((128, 128))).flatten() / 255.0

    # Load the trained model
    model = define_model()
    model.load_weights('fraud_image_model.h5')

    # Prepare the image for classification
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 1)

    # Use the trained model to predict
    prediction = model.predict(image)
    class_names = ['Fake Image', 'Real Image']
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class

# Define the Streamlit app
def main():
    st.title("Fraud Image Detection App")
    st.sidebar.header("Navigation")

    # Sidebar options
    selected_option = st.sidebar.selectbox("Choose an option", ["Home", "Image Classification", "Metadata Analysis", "About the Algorithm", "FAQ"])

    if selected_option == "Home":
        st.write("Welcome to the Fraud Image Detection App!")
        st.image("your_logo.png", use_column_width=True)
    
    elif selected_option == "Image Classification":
        st.header("Image Classification")
        image_path = st.file_uploader("Upload an image for classification", type=["jpg", "png"])
        if image_path:
            # Display the uploaded image
            uploaded_image = Image.open(image_path)
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Classify the image
            classification_result = classify_image(image_path)
            st.subheader("Image Classification Result:")
            st.write(classification_result)

    elif selected_option == "Metadata Analysis":
        st.header("Metadata Analysis")
        image_path = st.file_uploader("Upload an image for metadata analysis", type=["jpg", "png"])
        if image_path:
            # Display the uploaded image
            uploaded_image = Image.open(image_path)
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Perform metadata analysis
            metadata_details = perform_metadata_analysis(image_path)
            st.subheader("Metadata Analysis:")
            for key, value in metadata_details.items():
                st.write(f"{key}: {value}")

    elif selected_option == "About the Algorithm":
        st.header("About the Algorithm")
        st.write("This app uses a deep learning model to classify images as either real or fake.")
        st.write("The model has been trained on a dataset of real and fake images to make predictions.")
        st.write("It also performs metadata analysis to extract information about image properties.")

    elif selected_option == "FAQ":
        st.header("FAQ")
        st.write("**Q: How accurate is the image classification?**")
        st.write("A: The accuracy of the image classification depends on the quality of the dataset used for training.")
        st.write("**Q: What metadata is analyzed?**")
        st.write("A: The app analyzes metadata such as creation date/time, software used, and date modified.")
        st.write("**Q: Can I use this app for my own images?**")
        st.write("A: Yes, you can upload your own images for classification and metadata analysis.")

if __name__ == "__main__":
    main()
