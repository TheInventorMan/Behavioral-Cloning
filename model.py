import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import test_train_split

from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dropout, Lambda, Dense

def driveModel(input_shape=(66, 200, 3), optimizer="adam", loss="mse"):
    # Define our model :D
    model = Sequential()
    # Normalization layer
    model.add(Lambda(lambda x: (x/128)-1., input_shape=input_shape))
    
    # Convolutional layers
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2)))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))
    model.add(LeakyReLU(alpha=0.01))
    
    # Flatten to interface with FC layers
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.01))
    
    # Output
    model.add(Dense(1))
    
    # Compile!
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
   
    
    
    
    
    