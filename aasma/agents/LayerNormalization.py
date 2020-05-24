#!/usr/bin/env python3
#---------------------------------
# AASMA Single Thread
# File : DeepQ.py
#
# @ start date          21 05 2020
# @ last update         23 05 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

#---------------------------------
# class LayerNormalization
#---------------------------------
# Inspired by
# https://github.com/IntoxicatedDING/DQN-Beat-Atari/blob/master/dqn.py
# and ported from tf1 to tensorflow.keras (tf2)
class LayerNormalization(Layer):
    def __init__(self, eps=1e-5, activation=None, **kwargs):
        self.eps = eps
        self.channels = None
        self.activation = activation

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        shape = [1] * (len(input_shape) - 1)
        shape.append(self.channels)
        self.add_weight('gamma', shape, dtype='float32', initializer='ones')
        self.add_weight('beta', shape, dtype='float32', initializer='zeros')

        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dim = len(K.int_shape(inputs)) - 1
        mean = K.mean(inputs, axis=dim, keepdims=True)
        var = K.mean(K.square(inputs - mean), axis=dim, keepdims=True)
        outputs = (inputs - mean) / K.sqrt(var + self.eps)

        try:
            outputs = outputs * self.trainable_weights[0] + self.trainable_weights[1]
        except:
            pass

        if self.activation is None:
            return outputs
        else:
            return self.activation(outputs)

    def get_config(self):
        config = {
            'eps': self.eps,
            'channels': self.channels,
            'activation': self.activation
        }

        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
