
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

downloaded = drive.CreateFile({'id':"1NwrithRqi1lcpAPadM5Rz0AQrqmr-2DD"})
downloaded.GetContentFile('CatvsDogs.rar')

# !unrar x "/content/CatvsDogs.rar" "/content/"

import subprocess

# Path to the RAR file
rar_file = "path/to/CatvsDogs.rar"  # Replace with actual file path
output_dir = "path/to/extract_folder"  # Replace with the desired output folder path

# Extract using subprocess
subprocess.run(["unrar", "x", rar_file, output_dir], check=True)


import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,Sequential,layers,preprocessing
import os

file_names=os.listdir("/content/train")
dogorcat=[]
for name in file_names:
    category=name.split('.')[0]
    if category=='dog':
        dogorcat.append("DOG")
    else:
        dogorcat.append("CAT")
train=pd.DataFrame({
    'filename':file_names,
    'category':dogorcat
})

model=models.Sequential()
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
#model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.BatchNormalization())
#model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.BatchNormalization())
#model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])

training = preprocessing.image.ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
validation=preprocessing.image.ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)

trainingdata = training.flow_from_dataframe(train,"/content/train",x_col='filename',y_col='category',target_size=(64,64),class_mode='categorical')

model.fit(trainingdata,epochs=2)

from google.colab import drive
drive.mount('/content/drive')
os.makedirs("drive/My Drive/Models",exist_ok=True)
model.save("drive/My Drive/Models/my_model.keras") # Add .keras extension to the filename

!pip install streamlit
from PIL import Image,ImageOps
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np

# Remove the outdated configuration setting
# st.set_option('deprecation.showfileUploaderEncoding', False)

def load_models(img):
    model = models.load_model('D:/Programs/ML1/ML2/Assignment3/')
    image=img.resize((64,64))
    image_array=np.array(image)
    #image_array=tf.image.rgb_to_grayscale(image_array)
    image_array=(tf.reshape(image_array,(image_array.shape[0],image_array.shape[0],3)))/255 # Fixed indentation here
    image_array=np.array([image_array])

    prediction = model.predict_classes(image_array)
    return prediction

def upload_images():
    uploaded_file = st.file_uploader("Choose an Image ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded The image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = load_models(image)

        if label==0:
            st.write("It is a CAT")
        if label==1:
            st.write("It is a Dog")

if __name__ =="__main__":
    st.header("CAT AND DOG PREDICTION")
    st.write("Upload an image")

    upload_images()