#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we construct the first shot of cafe
"""
import os
# setting GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set gpu growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from config import *
from utils import *


def cafe_middle_output_gradient(optimizer, dummy_middle_output_gradient,random_lists, true_gradient):
    """
    In this function we implement the middle output gradient
    :return:
    """
    '''
    # Check the sum(middle output gradient) = bias gradient
    for worker_index in range(number_of_workers):
        temp_result = tf.reduce_sum(middle_output_gradient[worker_index], axis = 0) - true_gradient[worker_index+1][5]
        print(temp_result.numpy())
    '''
    batch_dummy_middle_output_gradient = take_batch(number_of_workers, dummy_middle_output_gradient, random_lists)
    middle_output_gradient_gradient = []
    with tf.GradientTape(persistent=True) as tape:
        for worker_index in range(number_of_workers):
            loss = tf.norm(tf.reduce_sum(batch_dummy_middle_output_gradient[worker_index], axis = 0) -
                           true_gradient[worker_index+1][5])**2
            temp_middle_output_gradient_gradient = tape.gradient(loss, batch_dummy_middle_output_gradient[worker_index])
            middle_output_gradient_gradient.append(temp_middle_output_gradient_gradient)
        print('first_shot_loss:', loss.numpy(), end='\t')
    optimizer.apply_gradients(zip(middle_output_gradient_gradient, batch_dummy_middle_output_gradient))
    dummy_middle_output_gradient_new = assign_to_dummy(
        number_of_workers, batch_size, dummy_middle_output_gradient,batch_dummy_middle_output_gradient, random_lists)
    """
    for worker_index in range(number_of_workers):
        np.save('dummy_middle_output_gradient_' + str(worker_index) + '.npy',
                dummy_middle_output_gradient[worker_index].numpy())
    """
    return dummy_middle_output_gradient_new

