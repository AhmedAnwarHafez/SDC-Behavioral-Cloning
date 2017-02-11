import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import csv
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# set the directory where the images and the driving log are stored in
track1_directory = "../data/track1-data/"
track1_log_directory = track1_directory + "driving_log.csv"
track1_lines = []

# read all driving log entries in a list
with open(track1_log_directory,'rt') as f:
    reader = csv.reader(f)
    for line in reader:
        track1_lines.append(line)
# take off the first row because it contains the headers
_ = track1_lines.pop(0)

# these two values are for the resizng input images
new_col, new_row = 16, 64

def process(img):
    ## Crop upper and lower parts and keep the middle
    img = img[70:140,:,:]
   
    ## resize
    img = cv2.resize(img, (new_row, new_col), interpolation = cv2.INTER_AREA)
    
    # convert the image from RGB to HSV to reduce memory usage
    colors.rgb_to_hsv(img)
    
    # input images will be grayscale by take the V channel only
    img = img[:,:,2]
    return img

def load_data(directory, lines):
    X = []
    y = []
    
    # loop through each row in the driving_log.csv
    for i in tqdm(range(len(lines))): 
        angle = float(lines[i][3])
        
        # load the actual images into memory
        center_img = plt.imread(directory + lines[i][0].strip())
        left_img = plt.imread(directory + lines[i][1].strip())
        right_img = plt.imread(directory + lines[i][2].strip())  

        # resize, crop and convert into grayscale          
        left_img = process(left_img)
        right_img = process(right_img)
        center_img =  process(center_img)

        X.append(center_img)
        y.append(angle)
        X.append(left_img)
        # add a shifting angle
        y.append(angle + 0.25)
        X.append(right_img)
        y.append(angle - 0.25)

    return np.array(X), np.array(y, np.float32)

# load all the images and angles in X_train and y_train
X_train, y_train = load_data(track1_directory, track1_lines)
X_train, y_train = shuffle(X_train, y_train)


# create 200 bins
bins = np.linspace(-1.0, 1.0, 200)

# allocate angles to each bin
digitized = np.digitize(y_train, bins)

y_filtered = []
X_filtered = []

# the number of examples to be taken from each bin
target = 100

for i in range(0, len(bins)):
    X_filtered.extend(X_train[digitized == i][:target])
    y_filtered.extend(y_train[digitized == i][:target])
    
y_filtered = np.array(y_filtered) 
X_filtered = np.array(X_filtered) 

# set the training set to the balanced data set
X_train = X_filtered
y_train = y_filtered

# augment the data by flipping the images and steering angle
X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
y_train = np.append(y_train,-y_train,axis=0)

# reshape the images so that original shape changes from (X,X,X) to (X,X,X,1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)


from keras.models import *
from keras.layers import *
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dense, Convolution2D, ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import json

model = Sequential()
# for faster convergence the input data is normalized by dividing by 127.5 and subtracting 1
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,64,1), output_shape=(16,64,1)))

model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.summary()

# the MSE is used because this a regression problem
model.compile(loss='mean_squared_error',
              optimizer='adam')

print(X_train.shape)
print(y_train.shape)

print("starting training model")
history = model.fit(X_train, y_train,
                    shuffle=True,
                    batch_size=32,
                    nb_epoch= 7, 
                    validation_split=0.1,
                    verbose=1)

filepath = "./model.h5"
model.save(filepath)