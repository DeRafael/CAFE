#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we construct double shot
"""

from config import *
from utils import *

def cafe_middle_input(
        optimizer, dummy_middle_output_gradient, dummy_middle_input, random_lists, true_gradient, real_middle_input,
        iter):
    """
    In this function we implement the middle output gradient
    :return:
    """
    batch_dummy_middle_input = take_batch(number_of_workers, dummy_middle_input, random_lists)
    batch_recovered_middle_output_gradient = take_batch(number_of_workers, dummy_middle_output_gradient, random_lists)
    '''
    # Check the sum(middle output gradient) = bias gradient
    for worker_index in range(number_of_workers):
        temp_result = tf.reduce_sum(middle_output_gradient[worker_index], axis = 0) - true_gradient[worker_index+1][5]
        print(tf.reduce_sum(temp_result.numpy()))
    # Check parameter = transpose(middle input) * middle_output_gradient
    for worker_index in range(number_of_workers):
        temp_result = true_gradient[worker_index+1][4] - tf.matmul(
        tf.transpose(real_middle_input[worker_index]),batch_dummy_middle_output_gradient[worker_index])
        print(tf.norm(temp_result.numpy())**2)
    '''
    middle_input_gradient = []
    with tf.GradientTape(persistent=True) as tape:
        for worker_index in range(number_of_workers):
            loss = tf.norm(true_gradient[worker_index+1][4] - tf.matmul(
                tf.transpose(batch_dummy_middle_input[worker_index]),
                batch_recovered_middle_output_gradient[worker_index]))**2
            temp_middle_input_gradient = tape.gradient(loss, batch_dummy_middle_input[worker_index])
            middle_input_gradient.append(temp_middle_input_gradient)
            MSE = tf.reduce_mean(
                tf.keras.losses.MSE(real_middle_input[worker_index], batch_dummy_middle_input[worker_index]))
        print('double shot loss:', loss.numpy(), 'MSE:' ,MSE.numpy(), end='\t')
    batch_dummy_middle_input = optimizer.apply_gradients(
        iter, batch_size, random_lists, middle_input_gradient, batch_dummy_middle_input)
    dummy_middle_input_new = assign_to_dummy(
        number_of_workers, batch_size, dummy_middle_input, batch_dummy_middle_input, random_lists)
    """
    for worker_index in range(number_of_workers):
        np.save('dummy_middle_input_' + str(worker_index) + '.npy', dummy_middle_input[worker_index].numpy())
    """
    return dummy_middle_input_new