# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:37:02 2019

@author: Ruihao Wang
"""

import os, cv2, random
import numpy as np
import pandas as pd

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
# import seaborn as sns
##matplotlib inline 

from tensorflow.keras import backend as K
# K.tensorflow_backend._get_available_gpus()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

import os, cv2, random
import numpy as np
import pandas as pd

ROWS = 256
COLS = 256
ROWS2 = 64
COLS2 = 64
CHANNELS = 3

cwd = os.getcwd()

#TRAIN_DIR = cwd + '/train/train/'
#
#TEST_DIR = cwd + '/test1/test1/'
TRAIN_DOG = cwd + '/train/dog/'
TRAIN_CAT = cwd + '/train/cat/'
TRAIN_HORSE = cwd + '/train/horse/'
TRAIN_BIRD = cwd + '/train/bird/'

#TEST_DIR = cwd + '/test/'

#train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if ] # use this for full dataset

train_dogs =   [TRAIN_DOG+i for i in os.listdir(TRAIN_DOG) if '.jpg' or '.jpeg' in i]
train_cats =   [TRAIN_CAT+i for i in os.listdir(TRAIN_CAT) if '.jpg' or '.jpeg' in i]
train_horses = [TRAIN_HORSE+i for i in os.listdir(TRAIN_HORSE) if '.jpg' or '.jpeg' in i]
train_birds =  [TRAIN_BIRD+i for i in os.listdir(TRAIN_BIRD) if '.jpg' or '.jpeg' in i]

print(train_dogs[1:5])
print(train_cats[1:5])
print(train_horses[1:5])
print(train_birds[1:5])

#train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog'and 'jpg' in i]
#train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat'and 'jpg' in i]

#test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

len_train = [len(train_dogs),len(train_cats),len(train_horses),len(train_birds)]
print(len_train)
#len_test = len(test_images)

# test_order = []
# for elem in test_images:
#     print(elem)
#     test_order.append(int(elem[-5]))



#print(test_images)
#print(test_images.shape)
# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
#print(train_dogs[:10])
#print(train_cats[:10])

random.shuffle(train_dogs)
random.shuffle(train_cats)
random.shuffle(train_horses)
random.shuffle(train_birds)
train_dogs = train_dogs
train_cats = train_cats
train_horses = train_horses
train_birds = train_birds
train_images = train_dogs + train_cats + train_horses + train_birds

print(len(train_dogs))
print(len(train_cats))
print(len(train_horses))
print(len(train_birds))
print(len(train_images))
#print(len_test)

random.shuffle(train_images)
#random.shuffle(test_images)

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR
    # print(img.shape)
    # b,g,r = cv2.split(img)
    # img2 = cv2.merge([r,g,b])
    # print(img2.shape)
    # imgplot = plt.imshow(img2)
    # plt.show()
    resized = cv2.resize(img, (ROWS2, COLS2), interpolation=cv2.INTER_CUBIC)
    #resized = cv2.resize(img2, dim, interpolation=cv2.INTER_CUBIC)
    #print resized.shape
    #cv2.imshow('resized',resized)
    #cv2.waitKey(0)

    resized = np.expand_dims(resized,axis = 2)
    #print resized.shape
    return resized

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS2, COLS2, 1), dtype=np.uint8)
    for i, image_file in enumerate(images):
        if i%1000 == 0:
            print('Processed {} of {}'.format(i, count))
            print(image_file)
        image = read_image(image_file)
        data[i] = image

    return data

print('Preparing training data')
train = prep_data(train_images)
print(train.shape)
#print('Preparing test data')
#test = prep_data(test_images[1:100])
#print(test[0].T.shape)
print("Train shape: {}".format(train.shape))
#print("Test shape: {}".format(test.shape))

#labels = []
#k = 0
#for i in train_images:
#    if k<len(train_dogs):
#        labels.append([1,0,0,0])
#    else if k<(len(train_dogs)+len(train_cats)):
#        labels.append([0,1,0,0])
#    else if k<(len(train_dogs)+len(train_cats)+len(train_horses)):
#        labels.append([0,0,1,0])
#    else if k<(len(train_dogs)+len(train_cats)+len(train_horses)len(train_birds)):
#        labels.append([0,0,0,1])
#    k = k + 1

#labels_dog = []
#labels_cat = []
#labels_horse = []
#labels_birds = []
#for i in train_dogs:
#    labels_dog.append([1,0,0,0])
#for i in train_cats:
#    labels_cat.append([0,1,0,0])
#for i in train_horses:
#    labels_horse.append([0,0,1,0])
#for i in train_birds:
#    labels_birds.append([0,0,0,1])

labels = []
for i in train_images:
    if  'dog' in i:
        labels.append([1,0,0,0])
    elif 'cat' in i:
        labels.append([0,1,0,0])
    elif 'horse' in i:
        labels.append([0,0,1,0])
    elif 'bird' in i:
        labels.append([0,0,0,1])
    
labels = np.array(labels)

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

def catdog():
    model = Sequential()

    model.add(Conv2D(16, 3, padding='same', input_shape= (64,64,1), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,3,padding = 'same',activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, padding='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

#    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    
    model.add(GlobalAveragePooling2D(data_format = 'channels_last'))
    
#    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.5))

    # model.add(GlobalAveragePooling2D(data_format = 'channels_last'))
    
    model.add(Dense(4))
    model.add(Activation('softmax'))
    print("Compiling model...")
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

print("Creating model:")
model = catdog()

epochs = 200
batch_size = 200

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto',restore_best_weights=True)
       

def run_catdog():
    
    history = LossHistory()
    print("running model...")
    model.fit(train, labels, batch_size=batch_size, epochs=epochs,
              validation_split=0.20, verbose=2, shuffle=True, callbacks=[history, early_stopping])
    
    # print("making predictions on test set...")
    # predictions = model.predict(test.astype(float), verbose=0)
    # return predictions, history
    return history

# predictions, history = run_catdog()
history = run_catdog()

loss = history.losses
val_loss = history.val_losses

######prediction some with figures

# for i in range(0,50):
#     if predictions[i, 0] >= 0.5:
#         print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
#     else:
#         print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))
#     plt.imshow(test[i].T[0,:,:])
#     plt.show()



#####predict cat | predict dog
# np.save('prediction.npy',predictions)
# np.savetxt('prediction.txt',predictions[:,0])

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cat-Dog Demo Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,epochs)[0::2])
plt.legend()
plt.show()

model.save('model_weight.h5')
model.save_weights('model.h5')
