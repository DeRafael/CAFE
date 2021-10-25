#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we construct the model
"""
import tensorflow as tf

class local_embedding(tf.keras.Model):
    def __init__(self, seed=1):
        super(local_embedding, self).__init__()
        tf.random.set_seed(seed)
        # convolutional layer 1
        self.c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', name='c1')
        # max pooling 1
        self.s1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='m1')
        # convolutional layer 2
        self.c2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', name='c2')
        # max pooling 2
        self.s2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='m2')
        # convolutional layer 3
        # self.c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', name='c3')
        # max pooling 3
        # self.s3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='m3')
        # activation function
        self.relu = tf.keras.activations.relu
        self.sigmoid = tf.keras.activations.sigmoid
        self.fc1 = tf.keras.layers.Dense(256, activation = None, name = 'fc1')
        self.fc2 = tf.keras.layers.Dense(64, activation = None, name = 'fc2')
        self.fc3 = tf.keras.layers.Dense(10, activation = None, name = 'fc3')

    def forward(self, input):
        # input size should be 14x14x3
        # print(input)
        x = tf.reshape(input, [-1, 14, 14, 1])
        x = self.c1(x)
        x = self.relu(x)
        x = self.s1(x)
        # x should be 7x7x64

        x = self.c2(x)
        x = self.relu(x)
        x = self.s2(x)
        # x should be 4x4x128

        '''
        x = self.c3(x)
        x = self.relu(x)
        x = self.s3(x)
        # x should be 6x6x64
        '''

        middle_input = tf.reshape(x, [-1, 4 * 4 * 128]) # 2304

        middle_output = self.fc1(middle_input)
        x = self.relu(middle_output)
        # x = self.sigmoid(middle_output)
        # x should be 256
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.sigmoid(x)
        # x should be 64
        x = self.fc3(x)
        x = self.relu(x)
        # x = self.sigmoid(x)
        return middle_input, x, middle_output

class server(tf.keras.Model):
    def __init__(self, seed=0):
        super(server, self).__init__()
        tf.random.set_seed(seed)
        self.last = tf.keras.layers.Dense(10, activation = 'softmax', name = 'last')

    def forward(self, input):
        # the size of input should be 1024
        output = self.last(input)
        # the size of output is 10
        return output



