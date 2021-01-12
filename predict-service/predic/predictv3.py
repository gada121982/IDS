# PREDICT
from __future__ import print_function
import tensorflow as tf
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

from joblib import dump, load
from keras import backend as K
#import matplotlib
import os
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, Normalizer
import PIL
import shutil
import time
from PIL import Image
import shutil
import re
import math
import numpy as np
from tensorflow.keras.preprocessing import image
import numpy as np


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def GRU_Preprocessing(df):
    df = clean_dataset(df)
    scaler = Normalizer().fit(df)
    df = scaler.transform(df)
    df = np.reshape(df, (df.shape[0], 1, df.shape[1]))
    return df


def CNN_Preprocessing(df):
    df = clean_dataset(df)
    scaler = Normalizer().fit(df)
    df = scaler.transform(df)
    df = np.reshape(df, (df.shape[0], df.shape[1], 1))
    return df


def RF_Preprocessing(df):
    df = clean_dataset(df)
    scaler = Normalizer().fit(df)
    df = scaler.transform(df)
    return df


model_GRU_path = '/home/truc/doan/server/predict-service/GRU.h5'
model_CNN_path = '/home/truc/doan/server/predict-service/cnn.h5'
model_RF_path = '/home/truc/doan/server/predict-service/RandomForest.joblib'


def predict(features_array):
    df = pd.DataFrame([features_array])
    count_True = 0
    count_False = 0

# Predict with GRU
    model_GRU = tf.keras.models.load_model(model_GRU_path)
    X = GRU_Preprocessing(df)
    result_GRU = model_GRU.predict_classes(X)
    if (result_GRU[0][0] == 0):
        count_False = count_False + 1
    else:
        count_True = count_True + 1

# Predict with CNN
    model_CNN = tf.keras.models.load_model(model_CNN_path)
    X = CNN_Preprocessing(df)
    result_CNN = model_CNN.predict_classes(X)
    if (result_CNN[0][0] == 0):
        count_False = count_False + 1
    else:
        count_True = count_True + 1

# Predict with RF
    model_RF = load(model_RF_path)
    X = RF_Preprocessing(df)
    result_RF = model_RF.predict(X)
    if (result_RF[0] == 0):
        count_False = count_False + 1
    else:
        count_True = count_True + 1

    if (count_True >= count_False):
        return True
    else:
        return False


features_array = [443,	6,	141385,	9,	7,	553,	3773.0,	30597.30523,
                  113.166177,	141385.0,	51417.0,	0,	0,	0,	0,	192,
                  152,	63.655975,	49.510203,	225352.389700,	0,	0,
                  1,	1,	0,	0,	0,	1,	0,	270.375000,	61.444444,	539.000000,
                  0,	9,	553,	7,	3773,	8192,	119,	4]

features_arrayAttack = [8080,	6,	16079,	3,	4,	326,	129.0,	28297.77971,
                        435.350457,	645.0,	15513.0,	0,	0,	0,	0,	72,	92,
                        186.578767,	248.771690,	13318.69643,	0,	0,	1,	1,	0,	0,	0,
                        1,	1,	65.0,	108.666667,	32.25,	0,	3,	326,	4,	129,	8192,	219,	1]

result = predict(features_arrayAttack)
print(result)
