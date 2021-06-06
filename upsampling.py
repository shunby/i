#type:ignore
from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.ops.gen_array_ops import pad
import math
import tensorflow_addons as tfa

class Upsampling():
    def __init__(self) -> None:
        pass
    
    def stretch2d(self, img, x_scale, y_scale):
        """
        In: (B,1,t,C)
        Out:(B,x_scale,t*y_scale,C)
        """
        x = tf.image.resize(img, [img.shape[1]*x_scale, img.shape[2]*y_scale], method="nearest")
        return x
    def upsample(self, c, scales, cin_time_length, cin_channels):
        """
        in: (B,t,C)
        out: (B,t*prod(scales),C)
        """
        c = layers.Conv1D(cin_channels, 1, use_bias=False)(c)
        c = layers.Reshape((1, cin_time_length, cin_channels))(c)
        for scale in scales:
            c = self.stretch2d(c, 1, scale)
            conv = layers.Conv2D(cin_channels, (scale*2-1,1), padding="same", use_bias=False, kernel_initializer=keras.initializers.Constant(value=1./(scale*2+1)))
            c = tfa.layers.WeightNormalization(conv)(c)
        c = layers.Reshape((-1, cin_channels))(c)
        return c
# i = keras.Input((3,3))
# j = Upsampling().upsample(i, [2,2], 3, 3)
# m = keras.Model(i,j)
# m.summary()
# print(m.predict(np.array([[[1,2,3],[4,5,6],[7,8,9]]])))