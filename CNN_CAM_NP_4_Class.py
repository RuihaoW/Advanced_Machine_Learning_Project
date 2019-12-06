# -*- coding: utf-8 -*-
"""
Advanced_Machine_Learning_Project

@author: Ruihao Wang, Yijie Zhou, Liujun Zhang, Yu Guo
"""

import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import RMSprop

ROWS = 256
COLS = 256
ROWS2 = 64
COLS2 = 64
CHANNELS = 3

cwd = os.getcwd()

TRAIN_DOG = cwd + '/train/dog/'
TRAIN_CAT = cwd + '/train/cat/'
TRAIN_HORSE = cwd + '/train/horse/'
TRAIN_BIRD = cwd + '/train/bird/'

train_dogs =   [TRAIN_DOG+i for i in os.listdir(TRAIN_DOG) if 'jpg' or 'jpeg' in i]
train_cats =   [TRAIN_CAT+i for i in os.listdir(TRAIN_CAT) if 'jpg' or 'jpeg' in i]
train_horses = [TRAIN_HORSE+i for i in os.listdir(TRAIN_HORSE) if 'jpg' or 'jpeg' in i]
train_birds =  [TRAIN_BIRD+i for i in os.listdir(TRAIN_BIRD) if 'jpg' or 'jpeg' in i]

len_train = [len(train_dogs),len(train_cats),len(train_horses),len(train_birds)]
print(len_train)

random.shuffle(train_dogs)
random.shuffle(train_cats)
random.shuffle(train_horses)
random.shuffle(train_birds)

train_dogs = train_dogs[1:2]
train_cats = train_cats[1:2]
train_horses = train_horses[1:2]
train_birds = train_birds[1:2]
train_images = train_dogs + train_cats + train_horses + train_birds

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR
    resized = cv2.resize(img, (ROWS2, COLS2), interpolation=cv2.INTER_CUBIC)
    resized = np.expand_dims(resized,axis = 2)
    return resized

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS2, COLS2, 1), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i%1000 == 0:
            print('Processed {} of {}'.format(i, count))
    return data

print('Preparing training data')
train = prep_data(train_images)
print(train.shape)
print("Train shape: {}".format(train.shape))


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
    model.add(Conv2D(32,3,padding = 'same',activation='relu'))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))   
    model.add(GlobalAveragePooling2D(data_format = 'channels_last'))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    print("Compiling model...")
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

print("Creating model:")
model = catdog()

epochs = 20
batch_size = 50

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
       

def run_catdog():
    history = LossHistory()
    print("running model...")
    model.fit(train, labels, batch_size=batch_size, epochs=epochs,
              validation_split=0.20, verbose=2, shuffle=True, callbacks=[history, early_stopping])
    return  history

history = run_catdog()

loss = history.losses
val_loss = history.val_losses


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
