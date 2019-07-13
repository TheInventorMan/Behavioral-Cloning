from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dropout, Lambda, Dense, Activation, BatchNormalization

def driveModel(input_shape=(100, 200, 3), optimizer="adam", loss="mse"):
    # Define our model :D
    model = Sequential()
    # Normalization layer
    model.add(BatchNormalization(input_shape=input_shape, axis=2))

    # Convolutional layers
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1)))
    model.add(Activation('relu'))

    # Flatten to interface with FC layers
    model.add(Flatten())

    # Dropout applied to convolutional stage output
    model.add(Dropout(0.5))

    # Fully connected layers
    model.add(Dense(100))
    model.add(Activation('relu'))

    # Extra dropout layer to prevent previous fc layer from "memorizing" the dataset
    model.add(Dropout(0.5))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    # Output
    model.add(Dense(1))

    # Compile!
    model.compile(loss=loss, optimizer=optimizer)

    return model
