#!/usr/bin/env python
# coding: utf-8

# In[10]:


from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np

def load_models(img): 
    model = models.load_model('/tmp/model.h5') 
    image = img.resize((28, 28)) 
    image_array = np.array(image) 
    image_array = tf.image.rgb_to_grayscale(image_array)
    image_array = tf.cast(image_array, tf.float32)  # Ensure dtype is float32
    image_array = image_array / 255.0
    image_array = tf.reshape(image_array, (image_array.shape[0], image_array.shape[1], 1))
    image_array = np.array([image_array])
    
    # Use model.predict and find the class with the highest probability
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=-1)  # Get the index of the max probability
    return predicted_class[0]  # Return the first element of the array

def upload_images(): 
    uploaded_file = st.file_uploader("Choose an Image ...", type="jpg") 
    if uploaded_file is not None: 
        image = Image.open(uploaded_file) 
        st.image(image, caption='Uploaded The image.', use_column_width=True) 
        st.write("Classifying...") 
        label = load_models(image) 
        st.write(f"It is a {label}")

if __name__ == "__main__": 
    st.header("MNIST DATA Classification") 
    st.write("Upload an image") 
    upload_images()


# In[ ]:




