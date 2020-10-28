# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:17:16 2020

@author: gilli
"""

#%%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

'''initialize directory for images'''

my_data_dir = 'cell_images'

print(os.listdir(my_data_dir))

test_path = my_data_dir+'\\test\\'
train_path = my_data_dir+'\\train\\'

print(os.listdir(test_path))

para_cell = train_path+'\\parasitized\\'+os.listdir(train_path+'\\parasitized')[0]
para_img= imread(para_cell)
plt.imshow(para_img)

print(para_img.shape)

unifected_cell_path = train_path+'\\uninfected\\'+os.listdir(train_path+'\\uninfected')[0]
unifected_cell = imread(unifected_cell_path)
plt.imshow(unifected_cell)

print(len(os.listdir(train_path+'\\parasitized')))
print(len(os.listdir(train_path+'\\uninfected')))
#%%
'''Here we determine the average dimensions of the cell images. We want all the data to be of the same dimension'''
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'\\uninfected'):
    
    img = imread(test_path+'\\uninfected'+'\\'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
    
sns.jointplot(dim1,dim2)  

'130'
print(np.mean(dim1)) 
'130'
print(np.mean(dim2))

image_shape = (130,130,3)

#%%

'''Using ImageDataGenerator, we can make our model more flexible to changing circumstances. This generator randomly alters the input data
within the chosen amounts for the available paramaters'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
plt.figure()
plt.imshow(para_img)
plt.figure()
plt.imshow(image_gen.random_transform(para_img))

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
 
 'CREATION OF MODEL'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

#https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary so we use sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''Early stopping function to prevent over fitting and to optimize computational time'''
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)

results = model.fit_generator(train_image_gen,epochs=20,
                              validation_data=test_image_gen,
                             callbacks=[early_stop])

'''from tensorflow.keras.models import load_model
model.save('malaria_detector.h5')'''

#%%

'''Evaluation of results'''
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

model.metrics_names
model.evaluate_generator(test_image_gen)

pred_probabilities = model.predict_generator(test_image_gen)
predictions = pred_probabilities > 0.5

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_image_gen.classes,predictions))
print(confusion_matrix(test_image_gen.classes,predictions))
