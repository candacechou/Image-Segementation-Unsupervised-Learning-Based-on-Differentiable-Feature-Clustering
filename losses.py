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


class loss():
    def __init__(self,mu,nCluster,Width,Height,scr=False,v=0,mask=0):
        self.mu         = mu
        self.v          = v
        self.scribble   = scr
        self.Width      = Width
        self.Height     = Height
        self.nCluster   = nCluster
        self.mask       = mask
    def losses(self,y_true, y_pred):
        #### similarity loss
        y_t           = tf.reshape(y_pred,[self.Width*self.Height,self.nCluster])
        ci            = tf.math.argmax(y_pred,axis = 3,output_type = tf.dtypes.int64, name=None)
        ci            = tf.reshape(ci,[self.Width*self.Height,1])
        scce          = tf.keras.losses.SparseCategoricalCrossentropy()#reduction=tf.keras.losses.Reduction.SUM)
        sim_loss      = scce(ci, y_t)
        sim_loss      = tf.cast(sim_loss, dtype='float64')
       
        #### spatial continuity loss
        outputHP      = tf.reshape(y_pred,(self.Width, self.Height, self.nCluster) )
        HPy           = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz           = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        HPy_target    = tf.zeros([self.Width-1,self.Height,self.nCluster],tf.float64)
        HPz_target    = tf.zeros([self.Width,self.Height-1,self.nCluster],tf.float64)
        mae           = tf.keras.losses.MeanAbsoluteError()
        lhpy          = mae(HPy,HPy_target)
        lhpz          = mae(HPz,HPz_target)
    
        con_loss      = lhpy + lhpz
#         print(con_loss.dtype)
        
        if self.scribble:
            ### scribble loss
            scr_loss  = scce(ci, y_t,sample_weight = tf.constant(self.mask))
            scr_loss  = tf.cast(scr_loss, dtype='float64')
#             scr_loss  = scr_loss * scr_mask
            
#             scr_loss = losses.categorical_crossentropy(y_true, y_pred)
            return sim_loss + self.mu * con_loss + self.v * scr_loss
        else:
            return sim_loss + self.mu * con_loss
        
#     def loss(self,I,J):
#         return self.losses(I,J)
            