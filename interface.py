import skeleton as sk 
import cv2
import streamlit as st
import os
import tempfile
import cv

# Load the images
image1 = cv2.imread('example_image.jpg')
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Assuming 'sk.colored_segmented_image' and 'cv.segmented_image' are in RGB format
image2 = cv2.cvtColor(sk.colored_segmented_image.astype('uint8'), cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(cv.segmented_image.astype('uint8'), cv2.COLOR_BGR2RGB)

# Display the images and their sizes in a row
st.header("YUKTIKA")
st.text("Image Segmentation Example")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(image1_rgb, width=200)
    st.text("Original Image")


with col2:
    st.image(image2, width=200)
    st.text("Segmented with Coreset")


with col3:       
    st.image(image3, width=200)
    st.text("Segmented without Coreset")


# Display relative error
st.header("Relative Error from Practical Coreset")
st.text(sk.relative_error_practicalCoreset)

st.header("Relative Error from Uniformly Random Coreset")
st.text(sk.relative_error_uniformCoreset)
