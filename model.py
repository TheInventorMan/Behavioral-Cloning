import csv
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from driveModel import driveModel
from keras.models import load_model
# Constants
LR_CORRECTION_FACTOR = 0.2
CSV_FILE_PATH = "data/driving_log.csv"
PROC_FILE_NAME = "data_processed/processed.txt"
REL_TO_IMG = "data/"
REL_TO_PROC = "data_processed/"
REL_TO_PROC_FLIPPED = "data_processed_f/"
BATCH_SIZE = 256
preprocess = True

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
    dataFile.close()

    return db
# Crop out landscape and car hood
def process_img(img):
    cropped_img = img[60:130, :]
    resized_img = cv2.resize(cropped_img, (200,100))
    return resized_img

def preprocessImgs():
    db = importData()
    processed_file = open(PROC_FILE_NAME, "w")
    c_ctr = 0
    l_ctr = 0
    r_ctr = 0
    for datapoint in db[:10]:
        steering_ang = float(datapoint[3][1:])
        try:
            c_fname = datapoint[0].strip()
            
            c = cv2.imread(REL_TO_IMG + c_fname)
            #print("ctype", type(c))
            c = process_img(c)
            cv2.imwrite(REL_TO_PROC + c_fname, c)
            entry = str(REL_TO_PROC + c_fname + " " + str(steering_ang) + "\n")
            processed_file.write(entry)

            c_f = cv2.flip(c, 1)
            cv2.imwrite(REL_TO_PROC_FLIPPED + c_fname, c_f)
            entry = str(REL_TO_PROC_FLIPPED + c_fname + " " + str(-steering_ang) + "\n")
            processed_file.write(entry)
        except:
            c_ctr += 1
            pass

        try:
            l = cv2.imread(REL_TO_IMG + datapoint[1].strip())
            #print("ltype", type(l))
            l = process_img(l)
            cv2.imwrite(REL_TO_PROC + datapoint[1], l)
            entry = str(REL_TO_PROC + datapoint[1] + " " + str(steering_ang + LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)

            l_f = cv2.flip(l, 1)
            cv2.imwrite(REL_TO_PROC_FLIPPED + datapoint[1], l_f)
            entry = str(REL_TO_PROC_FLIPPED + datapoint[1] + " " + str(-steering_ang - LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)
        except:
            l_ctr += 1


        try:
            r = cv2.imread(REL_TO_IMG + datapoint[2].strip())
            #print("rtype", type(l))
            r = process_img(r)
            cv2.imwrite(REL_TO_PROC + datapoint[2], r)
            entry = str(REL_TO_PROC + datapoint[2] + " " + str(steering_ang - LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)

            r_f = cv2.flip(r, 1)
            cv2.imwrite(REL_TO_PROC_FLIPPED + datapoint[2], r_f)
            entry = str(REL_TO_PROC_FLIPPED + datapoint[2] + " " + str(-steering_ang + LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)
        except:
            r_ctr += 1

    processed_file.close()


    print(len(db)*3)
    print("num center:", c_ctr)
    print("num right:", r_ctr)
    print("num left:", l_ctr)
    print(l_ctr+c_ctr+r_ctr)

    raise Exception("halt")
        # try:
        #     #print(datapoint)
        #     center_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[0]), cv2.COLOR_BGR2RGB)
        #     center_aug = process_img(center_img)
        #     center_aug_f = cv2.flip(center_aug, 1)
        #     X_train_batch.append(center_aug)
        #     Y_train_batch.append(steering_ang)
        #     X_train_batch.append(center_aug_f)
        #     Y_train_batch.append(-steering_ang)
        # except:
        #     pass
        #
        # try:
        #     left_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[1][1:]), cv2.COLOR_BGR2RGB)
        #     left_aug = process_img(left_img)
        #     left_aug_f = cv2.flip(left_aug, 1)
        #     X_train_batch.append(left_aug)
        #     Y_train_batch.append(steering_ang + LR_CORRECTION_FACTOR)
        #     X_train_batch.append(left_aug_f)
        #     Y_train_batch.append(-steering_ang - LR_CORRECTION_FACTOR)
        # except:
        #     pass
        #
        # try:
        #     right_img = cv2.cvtColor(cv2.imread(REL_TO_IMG + datapoint[2][1:]), cv2.COLOR_BGR2RGB)
        #     right_aug = process_img(right_img)
        #     right_aug_f = cv2.flip(right_aug, 1)
        #     X_train_batch.append(right_aug)
        #     Y_train_batch.append(steering_ang - LR_CORRECTION_FACTOR)
        #     X_train_batch.append(right_aug_f)
        #     Y_train_batch.append(-steering_ang + LR_CORRECTION_FACTOR)
        # except:
        #     pass #print("o noes")


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
        yield (np.array(X_valid_batch), np.array(Y_valid_batch))

if preprocess:
    preprocessImgs()
raise Exception("done")
# Main sequence
train_db = importData()
XY_train, XY_valid = train_test_split(train_db, test_size=0.1)
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
