import csv
import cv2
# Constants
LR_CORRECTION_FACTOR = 0.2
CSV_FILE_PATH = "data/driving_log.csv"
PROC_FILE_NAME = "data_processed/processed.txt"
REL_TO_IMG = "data/"
REL_TO_PROC = "data_processed/"
REL_TO_PROC_FLIPPED = "data_processed_f/"

# Crop out landscape and car hood
def process_img(img):
    cropped_img = img[60:130, :]
    resized_img = cv2.resize(cropped_img, (200,100))
    return resized_img

# One-time routine to save resized and augmented images in a separate folder
def preprocessImgs():
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
    # Make new file for prepared dataset
    processed_file = open(PROC_FILE_NAME, "w")
    c_ctr = 0
    l_ctr = 0
    r_ctr = 0
    ctr = 0
    for datapoint in db:
        steering_ang = float(datapoint[3][1:])
        try:
            c_fname = datapoint[0].strip()
            c = cv2.imread(REL_TO_IMG + c_fname)

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

        try:
            l_fname = datapoint[1].strip()
            l = cv2.imread(REL_TO_IMG + l_fname)

            l = process_img(l)
            cv2.imwrite(REL_TO_PROC + l_fname, l)
            entry = str(REL_TO_PROC + l_fname + " " + str(steering_ang + LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)

            l_f = cv2.flip(l, 1)
            cv2.imwrite(REL_TO_PROC_FLIPPED + l_fname, l_f)
            entry = str(REL_TO_PROC_FLIPPED + l_fname + " " + str(-steering_ang - LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)
        except:
            l_ctr += 1

        try:
            r_fname = datapoint[2].strip()
            r = cv2.imread(REL_TO_IMG + r_fname)

            r = process_img(r)
            cv2.imwrite(REL_TO_PROC + r_fname, r)
            entry = str(REL_TO_PROC + r_fname + " " + str(steering_ang - LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)

            r_f = cv2.flip(r, 1)
            cv2.imwrite(REL_TO_PROC_FLIPPED + r_fname, r_f)
            entry = str(REL_TO_PROC_FLIPPED + r_fname + " " + str(-steering_ang + LR_CORRECTION_FACTOR) + "\n")
            processed_file.write(entry)
        except:
            r_ctr += 1

        print(ctr)
        ctr += 1
        
    processed_file.close()

    # Turns out not all of the files in the .csv exist: 22% to be specific
    print(len(db)*3)
    print("num center:", c_ctr)
    print("num right:", r_ctr)
    print("num left:", l_ctr)
    print("dropped frames:", l_ctr+c_ctr+r_ctr)
    print("dropped pct:", 100*(l_ctr+c_ctr+r_ctr)/(len(db)*3))
