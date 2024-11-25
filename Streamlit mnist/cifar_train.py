#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import models, layers, datasets 

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() 
train_images = train_images / 255.0 
test_images = test_images / 255.0 

model = models.Sequential() 
model.add(layers.Flatten(input_shape=(32, 32, 3))) 
model.add(layers.Dense(512, activation="relu")) 
model.add(layers.Dense(128, activation="relu")) 
model.add(layers.Dense(32, activation="relu")) 
model.add(layers.Dense(10, activation="softmax")) 

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

model.save('/tmp/model1.h5')


# In[ ]:




