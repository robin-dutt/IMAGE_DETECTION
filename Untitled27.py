import streamlit as st
import cv2
import os

# Function to perform metadata analysis on an image using OpenCV
def perform_metadata_analysis(image_path):
    def extract_creation_datetime(image_path):
        creation_datetime = None
        try:
            image = cv2.imread(image_path)
            if image is not None:
                exif_data = image.getexif()
                if exif_data:
                    for tag, value in exif_data.items():
                        if tag == 0x9003:  # EXIF tag for DateTimeOriginal
                            creation_datetime = value
                            break
        except Exception as e:
            st.error(f"Error: {str(e)}")
        return creation_datetime

    metadata_details = extract_creation_datetime(image_path)
    return metadata_details

# Define the Streamlit app
def main():
    st.title("Image Analysis App")
    st.sidebar.header("Navigation")

    # Sidebar options
    selected_option = st.sidebar.selectbox("Choose an option", ["Home", "Image Classification", "Metadata Analysis", "About the Algorithm", "FAQ"])

    if selected_option == "Home":
        st.write("Welcome to the Image Analysis App!")

    elif selected_option == "Image Classification":
        st.header("Image Classification")
        st.write("Upload an image for classification:")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            # Perform image classification
            if st.button("Classify Image"):
                # Save the uploaded image to a temporary file
                temp_image_path = "temp_image.jpg"
                with open(temp_image_path, "wb") as temp_image_file:
                    temp_image_file.write(uploaded_image.read())

            # Perform classification on the uploaded image
            classification_result = classify_image(temp_image_path)

            st.write(f"Image classification result: {classification_result}")

            # Remove the temporary image file
            os.remove(temp_image_path)


    elif selected_option == "Metadata Analysis":
        st.header("Metadata Analysis")
        image_path = st.file_uploader("Upload an image for metadata analysis", type=["jpg", "png"])
        if image_path:
            # Display the uploaded image
            uploaded_image = cv2.imread(image_path)
            st.image(uploaded_image, channels="BGR", caption="Uploaded Image", use_column_width=True)
            
            # Perform metadata analysis
            metadata_details = perform_metadata_analysis(image_path)
            st.subheader("Metadata Analysis:")
            if metadata_details:
                st.write(f"Creation Date/Time: {metadata_details}")
            else:
                st.warning("Metadata not found in the image.")

    elif selected_option == "About the Algorithm":
        st.header("About the Algorithm")
        st.write("This app uses OpenCV for metadata analysis and provides a placeholder section for image classification.")
        st.write("You can replace the placeholder with your image classification algorithm.")

    elif selected_option == "FAQ":
        st.header("FAQ")
        st.subheader("Q1: How do I upload an image for analysis?")
        st.write("A1: In the 'Image Classification' and 'Metadata Analysis' sections, you can use the file uploader to upload an image in JPG or PNG format.")
        
        st.subheader("Q2: What does the 'About the Algorithm' section contain?")
        st.write("A2: The 'About the Algorithm' section provides information about the technology and tools used in this app. You can add details about your image analysis algorithm here.")
        
        st.subheader("Q3: How can I replace the placeholder for image classification?")
        st.write("A3: You can replace the placeholder in the 'Image Classification' section with your own image classification code. You may need to import and use machine learning models or libraries for your specific classification task.")

if __name__ == "__main__":
    main()
