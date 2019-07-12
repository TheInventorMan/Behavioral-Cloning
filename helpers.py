# Helper functions
import csv
import cv2
import numpy as np

from sklearn.utils import shuffle

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
    cropped_img = img[60:130, :, :]
    resized_img = cv2.resize(cropped_img, (200,100))
    return resized_img

def training_generator(train_samples):
    while True:
        samples = shuffle(train_samples)
        for idx in range(0, len(samples), BATCH_SIZE):
            batch = samples[idx:idx+BATCH_SIZE]
            X_train_batch = []
            Y_train_batch = []
            for datapoint in batch:
                center_img = cv2.imread(REL_TO_IMG + datapoint[0])
                left_img = cv2.imread(REL_TO_IMG + datapoint[1])
                right_img = cv2.imread(REL_TO_IMG + datapoint[2])
                steering_ang = datapoint[3]

                center_aug = process_img(center_img)
                left_aug = process_img(left_img)
                right_aug = process_img(right_img)

                center_aug_f = cv2.flip(center_aug, 1)
                left_aug_f = cv2.flip(left_aug, 1)
                right_aug_f = cv2.flip(right_aug, 1)

                X_train_batch.append(center_aug)
                Y_train_batch.append(steering_ang)
                X_train_batch.append(center_aug_f)
                Y_train_batch.append(-steering_ang)

                X_train_batch.append(left_aug)
                Y_train_batch.append(steering_ang + LR_CORRECTION_FACTOR)
                X_train_batch.append(left_aug_f)
                Y_train_batch.append(-steering_ang - LR_CORRECTION_FACTOR)

                X_train_batch.append(right_aug)
                Y_train_batch.append(steering_ang - LR_CORRECTION_FACTOR)
                X_train_batch.append(right_aug_f)
                Y_train_batch.append(-steering_ang + LR_CORRECTION_FACTOR)

        yield np.array(X_train_batch), np.array(Y_train_batch)

def valid_generator(valid_samples):
    while True:
        samples = shuffle(valid_samples)
        for idx in range(0, len(samples), BATCH_SIZE):
            batch = samples[idx:idx+BATCH_SIZE]
            X_valid_batch = []
            Y_valid_batch = []
            for datapoint in batch:
                center_img = cv2.imread(REL_TO_IMG + datapoint[0])
                left_img = cv2.imread(REL_TO_IMG + datapoint[1])
                right_img = cv2.imread(REL_TO_IMG + datapoint[2])
                steering_ang = datapoint[3]

                center_aug = process_img(center_img)
                left_aug = process_img(left_img)
                right_aug = process_img(right_img)

                center_aug_f = cv2.flip(center_aug, 1)
                left_aug_f = cv2.flip(left_aug, 1)
                right_aug_f = cv2.flip(right_aug, 1)

                X_valid_batch.append(center_aug)
                Y_valid_batch.append(steering_ang)
                X_valid_batch.append(center_aug_f)
                Y_valid_batch.append(-steering_ang)

                X_valid_batch.append(left_aug)
                Y_valid_batch.append(steering_ang + LR_CORRECTION_FACTOR)
                X_valid_batch.append(left_aug_f)
                Y_valid_batch.append(-steering_ang - LR_CORRECTION_FACTOR)

                X_valid_batch.append(right_aug)
                Y_valid_batch.append(steering_ang - LR_CORRECTION_FACTOR)
                X_valid_batch.append(right_aug_f)
                Y_valid_batch.append(-steering_ang + LR_CORRECTION_FACTOR)

        yield X_valid_batch, Y_valid_batch
