import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
#import keras_cv
import numpy as np
import pandas as pd
import random
import time
import warnings
from sklearn.utils import check_random_state
warnings.filterwarnings("ignore")
from numpy import convolve
from scipy.signal import convolve2d
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras

def set_global_seed(seed=343): # 343 is the seed we've been using everywhere else
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Make GPU deterministic (optional but recommended)
    #os.environ["TF_DETERMINISTIC_OPS"] = "1"


# Based on Royas Image Classfication architecture!s
def create_CNN_model(dropout_rate, NumOutput, original_height, original_width, filters=32, kernel_size=(2,2), pool_1=(2,1), pool_2=(2,2),  optimizer="adam", lr=1e-3):
    #############
    CNN_model = Sequential()
    CNN_model.add(Conv2D(filters, kernel_size=kernel_size, input_shape=(original_height, original_width, 1), activation='relu'))
    #CNN_model.add(BatchNormalization())
    CNN_model.add(MaxPooling2D(pool_size=pool_1, strides=2))
    CNN_model.add(Conv2D(filters, kernel_size=kernel_size, activation='relu'))
    #CNN_model.add(BatchNormalization())
    CNN_model.add(MaxPooling2D(pool_size=pool_2, strides=2))
    CNN_model.add(Dropout(rate = dropout_rate))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(units=64, activation='relu'))
    CNN_model.add(Dropout(rate = dropout_rate))
    CNN_model.add(Dense(units= NumOutput, activation='softmax'))
    
    # Build optimizer with name + learning rate
    opt = tf.keras.optimizers.get({
        "class_name": optimizer,
        "config": {"learning_rate": lr}
    })

    CNN_model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return CNN_model