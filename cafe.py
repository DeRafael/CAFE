#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we update the previous code to make the program functional
"""

import os
# setting GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# set gpu growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from config import *
from data_preprocess import train_datasets as train_ds
from data_preprocess import test_datasets as test_ds
from model import local_embedding, server
from first_shot import cafe_middle_output_gradient
from double_shot import cafe_middle_input
from utils import *
import gc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def vfl_cafe():
    """
    In this function we implement the stochastic deep leakage from gradient
    :return:
    """
    # set learning rate as global
    global cafe_learning_rate

    # define models
    local_net = []
    for worker_index in range(number_of_workers):
        temp_net = local_embedding()
        local_net.append(temp_net)
    Server = server()

    # set optimizers
    optimizer_server = tf.keras.optimizers.Adam(learning_rate=learning_rate_fl)
    optimizers = []
    for worker_index in range(number_of_workers):
        optimizers.append(tf.keras.optimizers.Adam(learning_rate=learning_rate_fl))
    # optimizer3
    optimizer_cafe = Optimizer_for_cafe(number_of_workers, data_number, cafe_learning_rate)
    # set optimizer1
    optimizer1 = tf.keras.optimizers.SGD(learning_rate=learning_rate_first_shot)
    """Initialization middle output gradient"""
    dummy_middle_output_gradient = dummy_middle_output_gradient_init(number_of_workers, data_number, feature_space=256)
    # set optimizer2
    optimizer2 = Optimizer_for_middle_input(number_of_workers, data_number, learning_rate_double_shot, 2048)
    """Initialization middle input"""
    dummy_middle_input = dummy_middle_input_init(number_of_workers, data_number, feature_space=2048)
    '''collect all the real data'''
    real_data, real_labels = list_real_data(number_of_workers, train_ds, data_number)
    test_data, test_labels = list_real_data(number_of_workers, test_ds, test_data_number)
    """Initialization dummy data & labels"""
    dummy_data, dummy_labels = dummy_data_init(number_of_workers, data_number, pretrain=False, true_label=None)
    # clean the text file
    file = open(filename + '.txt', 'w')
    file.close()

    for iter in range(max_iters):
        # select index
        random_lists = select_index(iter, data_number, batch_size)
        # take gradients
        true_gradient, batch_real_data, real_middle_input, middle_output_gradient, train_loss, train_acc \
            = take_gradient(number_of_workers, random_lists, real_data, real_labels, local_net, Server)
        '''Inner loop: CAFE'''
        # clear memory
        tf.keras.backend.clear_session()
        # first shot
        dummy_middle_output_gradient = cafe_middle_output_gradient(
            optimizer1, dummy_middle_output_gradient, random_lists, true_gradient)
        # second shot
        dummy_middle_input = cafe_middle_input(
            optimizer2, dummy_middle_output_gradient, dummy_middle_input, random_lists, true_gradient,
            real_middle_input, iter)
        # third shot
        # take batch dummy data
        batch_dummy_data, batch_dummy_label = take_batch_data(number_of_workers, dummy_data, dummy_labels,random_lists)
        # take recovered batch
        batch_recovered_middle_input = tf.concat(take_batch(number_of_workers, dummy_middle_input, random_lists),axis=1)
        # compute gradient
        D, cafe_gradient_x, cafe_gradient_y = cafe(number_of_workers, batch_dummy_data, batch_dummy_label,
                                                   local_net, Server, true_gradient, batch_recovered_middle_input)
        # optimize dummy data & label
        batch_dummy_data = optimizer_cafe.apply_gradients_data(iter, random_lists, cafe_gradient_x, batch_dummy_data)
        batch_dummy_label = optimizer_cafe.apply_gradients_label(iter, random_lists, cafe_gradient_y, batch_dummy_label)
        # assign dummy data
        dummy_data = assign_data(number_of_workers, batch_size, dummy_data, batch_dummy_data, random_lists)
        dummy_labels = assign_label(batch_size, dummy_labels, batch_dummy_label, random_lists)
        psnr = PSNR(batch_real_data, batch_dummy_data)
        # print results
        print(D, iter, cafe_learning_rate, train_loss.numpy(), train_acc.numpy())
        # write down results
        if iter % 100 == 0:
            # test accuracy
            loss, test_acc = test(number_of_workers, test_data, test_labels, local_net, Server)
            record(filename, [D, psnr, iter, train_loss.numpy(), test_acc.numpy()])

        # learning rate decay
        if iter % iter_decay == iter_decay - 1:
            cafe_learning_rate = cafe_learning_rate * decay_ratio
            # change the learning rate in the optimizer
            optimizer_cafe.lr = cafe_learning_rate
        # learning rate warm up
        '''
        if iter % iter_warm_up == iter_warm_up - 1:
            optimizer_server.learning_rate = 0.5
            for worker_index in range(number_of_workers):
                optimizers[worker_index].learning_rate = 0.5
        '''
        # update server
        optimizer_server.apply_gradients(zip(true_gradient[0], Server.trainable_variables))
        for worker_index in range(number_of_workers):
            optimizers[worker_index].apply_gradients(zip(true_gradient[worker_index+1],
                                                         local_net[worker_index].trainable_variables))
    # save recovered data as images
    visual_data(real_data, True)
    visual_data(dummy_data, False)
    # save recovered data & labels as numpy
    save_data(dummy_data, False)
    save_data(dummy_labels, True)

if __name__ == "__main__":
    vfl_cafe()

