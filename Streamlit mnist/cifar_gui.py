#!/usr/bin/env python
# coding: utf-8

# In[8]:


from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np

def load_models(img): 
    model = models.load_model('/tmp/model1.h5') 
    image = img.resize((32, 32)) 
    image_array = np.array(image) 
    image_array = (tf.reshape(image_array, (image_array.shape[0], image_array.shape[0], 3))) / 255.0 
    image_array = np.array([image_array]) 
    prediction = model.predict_classes(image_array) 
    return prediction 

def upload_images(): 
    uploaded_file = st.file_uploader("Choose an Image ...", type="jpg") 
    if uploaded_file is not None: 
        image = Image.open(uploaded_file) 
        st.image(image, caption='Uploaded The image.', use_column_width=True) 
        st.write("Classifying...") 
        label = load_models(image) 
        classes = ["aeroplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        st.write(f"It is {classes[label[0]]}")

if __name__ == "__main__": 
    st.header("CIFAR DATA Classification (DENSE LAYERS)") 
    st.write("Upload an image") 
    upload_images()


# In[ ]:




