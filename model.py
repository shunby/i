# pyright: reportUnboundVariable=false
from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.ops.gen_array_ops import pad
import math
from upsampling import Upsampling

class model(object):
    def __init__(self, out_channels, time_length, residual_channels, residual_layers, residual_stacks,
            skip_out_channels, kernel_size, dropout, cin_channels, cin_time_length, gin_channels, gate_channels,
            num_speakers, use_gin, use_cin):
        self.out_channels = out_channels
        self.time_length = time_length
        self.residual_channels = residual_channels
        self.residual_layers = residual_layers
        self.residual_stacks = residual_stacks
        self.skip_out_channels = skip_out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.cin_channels = cin_channels
        self.gin_channels = gin_channels
        self.gate_channels = gate_channels
        self.num_speakers = num_speakers
        self.cin_time_length = cin_time_length
        self.use_gin = use_gin
        self.use_cin = use_cin
    
    def res_conv1d_glu(self, x, c, g, dilation, ind):
        split_former = layers.Lambda(lambda x : x[:, :,:x.shape[-1]//2], name=f"formerx{ind}")
        split_latter = layers.Lambda(lambda x : x[:, :,x.shape[-1]//2:], name=f"latterx{ind}")

        residual = x
        x = layers.Dropout(0.05)(x)
        x = layers.Conv1D(self.gate_channels, self.kernel_size, padding="causal", dilation_rate=dilation)(x)
        a = split_former(x)
        b = split_latter(x)

        if c is not None:
            c = layers.Conv1D(self.gate_channels, 1, use_bias=False)(c)
            split_former = layers.Lambda(lambda x : x[:, :,:x.shape[-1]//2], name=f"formerc{ind}")
            split_latter = layers.Lambda(lambda x : x[:, :,x.shape[-1]//2:], name=f"latterc{ind}")
            ca = split_former(c)
            cb = split_latter(c)
            a = layers.Add()([a,ca])
            b = layers.Add()([b,cb])
        if g is not None:
            g = layers.Conv1D(self.gate_channels, 1, use_bias=False)(g)
            split_former = layers.Lambda(lambda x : x[:, :,:x.shape[-1]//2], name=f"formerg{ind}")
            split_latter = layers.Lambda(lambda x : x[:, :,x.shape[-1]//2:], name=f"latterg{ind}")
            ga = split_former(g)
            gb = split_latter(g)
            a = layers.Add()([a,ga])
            b = layers.Add()([b,gb])
        x = layers.Multiply()([layers.Activation(activation="tanh")(a), layers.Activation(activation="sigmoid")(b)])
        s = Conv1D(self.skip_out_channels, 1)(x)
        x = Conv1D(self.residual_channels, 1)(x)
        x = layers.Add()([x, residual])
        x *=  math.sqrt(0.5)
        return x, s

    def wavenet(self):
        """
        Input: (B, T, C)
        g: (B,1)
        c: (B,*,*)
        """
        inputs = keras.Input(shape=(self.time_length, self.out_channels), name="audio")
        g = None
        c = None
        if self.use_gin:
            gin = keras.Input(shape=(1), name="global")
            g = layers.Embedding(self.num_speakers, self.gin_channels)(gin)
            g = layers.Reshape((1, self.gin_channels))(g)
            g = layers.Lambda(lambda x: tf.repeat(x, repeats=[self.time_length], axis=1))(g)
        if self.use_cin:
            cin = keras.Input(shape=(self.cin_time_length, self.cin_channels), name="local")
            c = Upsampling().upsample(cin, [4,4,4,4], self.cin_time_length, self.cin_channels)

        x = layers.Conv1D(self.residual_channels, 1, padding="causal")(inputs)
        for layer in range(self.residual_layers):
            dilation = 2 ** (layer % (self.residual_layers // self.residual_stacks))
            x, h = self.res_conv1d_glu(x, c, g, dilation, layer)
            if "skips" not in locals():
                skips = h
            else:
                skips = layers.Add()([skips,h])
        skips *= math.sqrt(1.0/self.residual_layers)
        x = skips
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(self.skip_out_channels, 1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(self.out_channels, 1)(x)
        x = layers.Activation("softmax")(x)
        input_tensors = [inputs]
        if self.use_gin:
            input_tensors.append(gin)
        if self.use_cin:
            input_tensors.append(cin)
        return keras.Model(input_tensors, x)
# m = model(100,4096,1000,3,1,1000,1,0.05,20,16,40,1000,10, False, False).wavenet()
# keras.utils.plot_model(m, to_file="model2.png")
# m.summary()