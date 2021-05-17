import tensorflow as tf
import cv2
import sys
import numpy as np
import random
from matplotlib import pyplot as plt


import keras
from keras import losses
from keras.models import Sequential,Model
from keras.layers.core import *
from keras.layers import*
from keras.optimizers import *
from keras.utils import to_categorical
from keras import backend as K



def MyNet(input_height, input_width,nChannel,nCluster,nCov):
    img_input = Input(shape=(input_height,input_width,3))
    conv1 = Conv2D( filters = nChannel,
                      kernel_size = (3, 3), 
                      activation='relu', 
                      strides=1,
                      padding = 'same')(img_input)

    conv1 = BatchNormalization()(conv1)

    for i in range(nCov-1):
        conv1 = Conv2D(filters = nChannel, 
                       kernel_size = (3, 3), 
                       activation='relu', 
                       strides=1,
                       padding = 'same')(conv1)
        conv1 = BatchNormalization()(conv1)
### 1D Conv

    conv2 = Conv2D(filters = nCluster, 
                   kernel_size = (1,1),
                   strides = 1,
                   padding = 'same')(conv1)
    conv2 = BatchNormalization()(conv2)
    model = Model(inputs=img_input, outputs = conv2)
    return model