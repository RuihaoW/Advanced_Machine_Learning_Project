"""
Advanced_Machine_Learning_Project
Ruihao Wang, Yijie Zhou, Liujun Zhang, Yu Guo
"""

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K 
import cv2



ROWS = 256
COLS = 256
ROWS2 = 64
COLS2 = 64
CHANNELS = 3

def main(image):
    model = load_model('model_weight.h5')
    model.summary()
    image_arr_2,image_arr = prep_data(image)
    image_arr = np.expand_dims(image_arr,axis=0)
    print('Input image array size')
    print(image_arr.shape)
    
    layer_4 = K.function([model.layers[0].input],[model.layers[4].output])
    f1 = layer_4([image_arr])[0]
    print ('The fifth pooling layer output')
    print(f1.shape)
    weights = model.layers[6].get_weights()
    weights = np.asarray(weights)[0]
    print(weights.shape)
    re_f1 = np.squeeze(f1,axis = 0)
    cam1 = np.zeros((64,64))
    cam2 = np.zeros((64,64))
    cam3 = np.zeros((64,64))
    cam4 = np.zeros((64,64))

    for i in range(f1.shape[3]):
        cam1 += re_f1[:,:,i]*weights[i][0]
        cam2 += re_f1[:,:,i]*weights[i][1]  
        cam3 += re_f1[:,:,i]*weights[i][2]
        cam4 += re_f1[:,:,i]*weights[i][3]  
    img_shape = cv2.imread(image, cv2.IMREAD_GRAYSCALE).shape
    print(img_shape)

    cam1 = cv2.resize(cam1,(64,64),interpolation = cv2.INTER_CUBIC)
    cam2 = cv2.resize(cam2,(64,64),interpolation = cv2.INTER_CUBIC)  
    print(cam1)
    print(cam1.shape)
    print(image_arr_2)
    spp1 = np.uint8(cam1) + image_arr_2
    spp2 = np.uint8(cam2) + image_arr_2
    spp3 = np.uint8(cam3) + image_arr_2
    spp4 = np.uint8(cam4) + image_arr_2

    cv2.imwrite('heatmap_for_dog.jpg',cam1)
    cv2.imwrite('heatmap_for_cat.jpg',cam2)
    cv2.imwrite('heatmap_for_horse.jpg',cam3)
    cv2.imwrite('heatmap_for_bird.jpg',cam4)
    cv2.imwrite('spp_for_cat.jpg',spp1)
    cv2.imwrite('spp_for_dog.jpg',spp2)
    cv2.imwrite('spp_for_horse.jpg',spp3)
    cv2.imwrite('spp_for_bird.jpg',spp4)
    np.save('heatmap_for_dog.npy',cam1)
    np.save('heatmap_for_cat.npy',cam2)
    np.save('heatmap_for_horse.npy',cam3)
    np.save('heatmap_for_bird.npy',cam4)
    predictions = model.predict(image_arr)
    print(predictions)
    predicted_class = np.argmax(predictions)
    print('The class is')
    if predicted_class == 0:
        print('Dog')
    elif predicted_class == 1:
        print('Cat')
    elif predicted_class == 2:
        print('Horse')    
    elif predicted_class == 3:
        print('Bird')

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COL
    print('The input image size')
    print(img.shape)
    resized = cv2.resize(img, (ROWS2, COLS2), interpolation=cv2.INTER_CUBIC)
    print ('The resized image size')
    print (resized.shape)
    resized_2 = np.expand_dims(resized,axis = 2)
    return resized,resized_2

def prep_data(images):
    data = np.ndarray(( ROWS2, COLS2,1), dtype=np.uint8)
    data = read_image(images)
    return data

if __name__ == '__main__':
    string = ".jpg"
    cwd = os.getcwd()
    for i in range(1):
        image = r'C:\Users\Ruihao Wang\PycharmProjects\ML\AML_Project\MaoGou_Demo\train\bird\129.jpeg'
        main(image)
