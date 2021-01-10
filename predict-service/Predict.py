from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
import matplotlib
import os
from sklearn.model_selection import train_test_split
import PIL
import shutil
import time
from PIL import Image
import shutil
import re
import math
from keras.preprocessing import image
import numpy as np


def float_bin(my_number, places=3):
    if "." in str(my_number):
        my_whole, my_dec = str(my_number).split(".")
        my_whole = int(my_whole)
        res = (str(bin(my_whole))+".").replace('0b', '')
    else:
        my_whole = int(my_number)
        my_dec = str(0)
        res = (str(bin(my_whole))+".").replace('0b', '')
    for x in range(places):
        my_dec = str('0.')+str(my_dec)
        temp = '%1.20f' % (float(my_dec)*2)
        my_whole, my_dec = temp.split(".")
        res += my_whole
    return res


def IEEE754(n):
    # identifying whether the number
    # is positive or negative
    sign = 0
    if n < 0:
        sign = 1
        n = n * (-1)
    p = 52
    # convert float to binary
    dec = float_bin(n, places=p)
    dotPlace = dec.find('.')
    onePlace = dec.find('1')
    # finding the mantissa
    if onePlace > dotPlace:
        dec = dec.replace(".", "")
        onePlace -= 1
        dotPlace -= 1
    elif onePlace < dotPlace:
        dec = dec.replace(".", "")
        dotPlace -= 1
    mantissa = dec[onePlace+1:]
    # calculating the exponent(E)
    exponent = dotPlace - onePlace
    exponent_bits = exponent + 1023
    # converting the exponent from
    # decimal to binary
    exponent_bits = bin(exponent_bits).replace("0b", '')
    mantissa = mantissa[0:52]
    # the IEEE754 notation in binary
    final = str(sign) + exponent_bits.zfill(11) + mantissa.zfill(52)
    # convert the binary to hexadecimal
    return (final)


def numeric_features(df):
    X = df
    frames = [X]
    result = pd.concat(frames)
    COLUMN_NAMES = ['feature', 'max', 'min']
    df = pd.DataFrame(columns=COLUMN_NAMES)
    for i in result.columns:
        if (result[i].dtype != 'object'):
            row = pd.DataFrame(
                [[i, result[i].max(), result[i].min()]], columns=COLUMN_NAMES)
            df = df.append(row)
    for i in X.columns:
        if X[i].dtype == 'float64':
            # print(i) #In ra cot minh dang lam
            X[i] = X[i].astype('float32')
            # Lay gia tri max cua features nay trong mang df
            max = df[df['feature'] == i]['max'].values[0]
            # Lay gia tri min cua features nay trong mang df
            min = df[df['feature'] == i]['min'].values[0]
            # Tinh toan la gia tri cua X[i] = (x-min)/max-min
            X[i] = X[i].apply(lambda x: (x-min)/(max-min))

    for i in X.columns:
        if (X[i].dtype == 'object'):
            # print(i)
            X[i] = X[i].astype('category').cat.codes
            X[i] = X[i].apply(lambda x: '{:08b}'.format(x))

    for i in X.columns:
        if (X[i].dtype == 'int64'):
            # print(i)
            X[i] = X[i].astype('int32')
            X[i] = X[i].apply(lambda x: '{:064b}'.format(x))

    for i in X.columns[X.isna().any()].tolist():
        X[i].fillna(-1.0, inplace=True)

    for i in X.columns:
        if (X[i].dtype == 'float64'):
            # print(i)
            X[i] = X[i].astype('float32')
            X[i] = X[i].apply(lambda x: IEEE754(x))
    return X


def convert_to_image(X):
    for index, row in X.iterrows():
        # nối chuỗi các selected features (từ 1 đến 42)
        joined_features = ''.join(
            row.values[0:len(X.columns)-1]).replace('\'', '')
        joined_features = joined_features.replace(
            "-", "",)  # Thay the dau - bang chuoi rong
        image_size = round(math.sqrt(len(joined_features)/8))+1
        image_size = 32
        print("image_size: ", image_size)
        arrayA = np.array(re.findall(
            '.{1,8}', joined_features.ljust(image_size*image_size*8, '0')))
        arrayA = vectorized_bin2dec(arrayA)
        arrayA = arrayA.reshape(image_size, image_size)
        arrayA = arrayA.astype(np.uint8)
        image = Image.fromarray(arrayA)
        rgbimg = Image.new("RGBA", image.size)
        rgbimg.paste(image)
        image = rgbimg
        return(image)


def bin2dec(i): return int(i, 2)


vectorized_bin2dec = np.vectorize(bin2dec)


def predict(model_VGG16_path, model_Resnet50_path, features_array):
    # Xử lý preprocessing_data
    x_1 = numeric_features(features_array)
    img = image.load_img(convert_to_image(x_1))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    model_res = tf.keras.models.load_model(model_Resnet50_path)
    model_vgg = tf.keras.models.load_model(model_VGG16_path)
    all_models = list()
    all_models.append(model_res)
    all_models.append(model_vgg)
    yhats = [model.predict(img) for model in all_models]
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)

    if (result == 0):
        return False
    else:
        return True


pathVGG = "/Users/hainguyen/dev/python/flask/ResNet50_model.h5"
path50 = "/Users/hainguyen/dev/python/flask/VGG16_model.h5"
features_array = [1, 2, 3, 4, 5, 5, 6, 7, 7, 8,
                  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
predict(pathVGG, path50, features_array)
