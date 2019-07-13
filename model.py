import csv
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from driveModel import driveModel
from preprocess import *
from keras.models import load_model
# Constants
PROC_FILE_NAME = "data_processed/processed.txt"
BATCH_SIZE = 256
preprocess = False # Flip this switch on to prepare dataset and save static copies to disk.

def importData():
    parsed_db = []
    db_file = open(PROC_FILE_NAME, 'r')

    for line in db_file:
        parsed_db.append(line.split())
    db_file.close()

    return parsed_db

def training_generator():
    global XY_train
    while True:
        samples = shuffle(XY_train)
        for idx in range(0, len(samples), BATCH_SIZE):
            batch = samples[idx:idx+BATCH_SIZE]
            X_train_batch = []
            Y_train_batch = []
            for datapoint in batch:
                steering_ang = float(datapoint[1])
                try:
                    img = cv2.cvtColor(cv2.imread(datapoint[0]), cv2.COLOR_BGR2RGB)
                    X_train_batch.append(img)
                    Y_train_batch.append(steering_ang)
                except:
                    pass

        yield (np.array(X_train_batch), np.array(Y_train_batch))

def valid_generator():
    global XY_valid
    while True:
        samples = shuffle(XY_valid)
        for idx in range(0, len(samples), BATCH_SIZE):
            batch = samples[idx:idx+BATCH_SIZE]
            X_valid_batch = []
            Y_valid_batch = []

            for datapoint in batch:
                steering_ang = float(datapoint[1])
                try:
                    img = cv2.cvtColor(cv2.imread(datapoint[0]), cv2.COLOR_BGR2RGB)
                    X_valid_batch.append(img)
                    Y_valid_batch.append(steering_ang)
                except:
                    pass
        yield (np.array(X_valid_batch), np.array(Y_valid_batch))

# Can optionally redo preprocessing with a single bool switch:
if preprocess:
    preprocessImgs()

# Import prepared dataset
train_db = importData()
# Make training and validation sets
XY_train, XY_valid = train_test_split(train_db, test_size=0.1)

# Trains model one epoch at a time, saving entire model to disk at each iteration.
for i in range(10):
    print("epoch", i)
    if (i==0):
        initialized_model = driveModel()
        initialized_model.save(str(i+1)+'model.h5')
        continue

    initialized_model = load_model(str(i) + 'model.h5')
    history_object = initialized_model.fit_generator(generator=training_generator(),
                                                 steps_per_epoch=np.ceil(2*len(XY_train)/BATCH_SIZE),
                                                 epochs=1,
                                                 verbose = 1,
                                                 validation_data=valid_generator(),
                                                 validation_steps=np.ceil(len(XY_valid)/BATCH_SIZE))
    initialized_model.save(str(i+1) + 'model.h5')
