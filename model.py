import csv
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from driveModel import driveModel
# Constants
LR_CORRECTION_FACTOR = 0.2
CSV_FILE_PATH = "data/driving_log.csv"
REL_TO_IMG = "data/"
BATCH_SIZE = 64

def importData():
    # Import csv data into db
    db = []

    dataFile = open(CSV_FILE_PATH)
    readingAgent = csv.reader(dataFile)
    next(readingAgent) #skip first line

    for line in readingAgent:
        entry = line[:-3] # get rid of non-steering data
        db.append(entry)
        #print(entry)
    return db

# Crop out landscape and car hood
def process_img(img):
    cropped_img = img[60:130, :]
    resized_img = cv2.resize(cropped_img, (200,100))
    return resized_img

def training_generator():
    global XY_train
    while True:
        samples = shuffle(XY_train)
        for idx in range(0, len(samples), BATCH_SIZE):
            batch = samples[idx:idx+BATCH_SIZE]
            X_train_batch = []
            Y_train_batch = []
            for datapoint in batch:
                steering_ang = float(datapoint[3][1:])
                try:
                    #print(datapoint)
                    center_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[0]), cv2.COLOR_BGR2RGB)
                    center_aug = process_img(center_img)
                    center_aug_f = cv2.flip(center_aug, 1)
                    X_train_batch.append(center_aug)
                    Y_train_batch.append(steering_ang)
                    X_train_batch.append(center_aug_f)
                    Y_train_batch.append(-steering_ang)
                except:
                    pass

                try:
                    left_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[1][1:]), cv2.COLOR_BGR2RGB)
                    left_aug = process_img(left_img)
                    left_aug_f = cv2.flip(left_aug, 1)
                    X_train_batch.append(left_aug)
                    Y_train_batch.append(steering_ang + LR_CORRECTION_FACTOR)
                    X_train_batch.append(left_aug_f)
                    Y_train_batch.append(-steering_ang - LR_CORRECTION_FACTOR)
                except:
                    pass

                try:
                    right_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[2][1:]), cv2.COLOR_BGR2RGB)
                    right_aug = process_img(right_img)
                    right_aug_f = cv2.flip(right_aug, 1)
                    X_train_batch.append(right_aug)
                    Y_train_batch.append(steering_ang - LR_CORRECTION_FACTOR)
                    X_train_batch.append(right_aug_f)
                    Y_train_batch.append(-steering_ang + LR_CORRECTION_FACTOR)
                except:
                    pass #print("o noes")
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
                steering_ang = float(datapoint[3][1:])
                try:
                    #print(datapoint)
                    center_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[0]), cv2.COLOR_BGR2RGB)
                    center_aug = process_img(center_img)
                    center_aug_f = cv2.flip(center_aug, 1)
                    X_valid_batch.append(center_aug)
                    Y_valid_batch.append(steering_ang)
                    X_valid_batch.append(center_aug_f)
                    Y_valid_batch.append(-steering_ang)
                except:
                    pass

                try:
                    left_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[1][1:]), cv2.COLOR_BGR2RGB)
                    left_aug = process_img(left_img)
                    left_aug_f = cv2.flip(left_aug, 1)
                    X_valid_batch.append(left_aug)
                    Y_valid_batch.append(steering_ang + LR_CORRECTION_FACTOR)
                    X_valid_batch.append(left_aug_f)
                    Y_valid_batch.append(-steering_ang - LR_CORRECTION_FACTOR)
                except:
                    pass

                try:
                    right_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[2][1:]), cv2.COLOR_BGR2RGB)
                    right_aug = process_img(right_img)
                    right_aug_f = cv2.flip(right_aug, 1)
                    X_valid_batch.append(right_aug)
                    Y_valid_batch.append(steering_ang - LR_CORRECTION_FACTOR)
                    X_valid_batch.append(right_aug_f)
                    Y_valid_batch.append(-steering_ang + LR_CORRECTION_FACTOR)
                except:
                    pass
        yield (X_valid_batch, Y_valid_batch)

# Main sequence
train_db = importData()
XY_train, XY_valid = train_test_split(train_db, test_size=0.05)
initialized_model = driveModel()
history_object = initialized_model.fit_generator(generator=training_generator(),
                                                 steps_per_epoch=np.ceil(len(XY_train)/BATCH_SIZE),
                                                 epochs=1,
                                                 verbose = 1,
                                                 validation_data=valid_generator(),
                                                 validation_steps=np.ceil(len(XY_valid)/BATCH_SIZE))
driveModel.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
