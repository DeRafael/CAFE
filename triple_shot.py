#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we update the previous code to make the program functional
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
from data_preprocess import train_datasets as train_ds
from model import local_embedding, server
from first_shot import cafe_middle_output_gradient
from double_shot import cafe_middle_input
from utils import *
import gc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# load data
# modelnet 40 3183 training images & 800 testing images
# take batch size
# train_ds = train_dataset.batch(train_image_count)
# test_ds = test_dataset.batch(test_image_count)

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
    optimizer_fl = tf.keras.optimizers.Adam(learning_rate = learning_rate_fl)
    optimizer_cafe = Optimizer_for_cafe(number_of_workers, data_number, cafe_learning_rate)
    '''collect all the real data'''
    real_data, real_labels = list_real_data(number_of_workers, train_ds, data_number)
    """Initialization dummy data & labels"""
    dummy_data, dummy_labels = dummy_data_init(number_of_workers, data_number, pretrain=False, true_label=None)
    # clean the text file
    file = open(filename + '.txt', 'w')
    file.close()

    '''The outer loop, currently max_iters=1, prepare for dynamic cases'''
    for iter in range(max_iters):
        '''The first shot: recover middle output gradient'''
        if first_shot_pretrain:
            # if pretrained just load
            dummy_middle_output_gradient = []
            for worker_index in range(number_of_workers):
                temp_middle_output_gradient = tf.convert_to_tensor(np.load(
                    'dummy_middle_output_gradient_' + str(worker_index) + '.npy'))
                dummy_middle_output_gradient.append(temp_middle_output_gradient)
        else:
            # if not pretrained
            dummy_middle_output_gradient = cafe_middle_output_gradient(real_data, real_labels, local_net, Server)

        '''Double shot: recover middle input'''
        if double_shot_pretrain:
            # load middle input
            dummy_middle_input = []
            for worker_index in range(number_of_workers):
                temp_middle_input = tf.convert_to_tensor(np.load('dummy_middle_input_' + str(worker_index) + '.npy'))
                dummy_middle_input.append(temp_middle_input)
        else:
            dummy_middle_input = cafe_middle_input(real_data, real_labels, local_net, Server,
                                                   dummy_middle_output_gradient)
        '''Inner loop: triple shot'''
        for cafe_iter in range(max_cafe_iters):
            # clear memory
            tf.keras.backend.clear_session()
            # select index
            random_lists = select_index(cafe_iter, data_number, batch_size)
            # take gradients
            true_gradient, batch_real_data, real_middle_input, middle_output_gradient = take_gradient(
                number_of_workers, random_lists,real_data, real_labels, local_net, Server)
            # take batch dummy data
            batch_dummy_data, batch_dummy_label = take_batch_data(number_of_workers, dummy_data, dummy_labels,
                                                                  random_lists)
            # take recovered batch
            batch_recovered_middle_input = tf.concat(take_batch(number_of_workers, dummy_middle_input, random_lists),
                                                     axis=1)
            # compute gradient
            D, cafe_gradient_x, cafe_gradient_y = cafe(number_of_workers, batch_dummy_data, batch_dummy_label,
                                                       local_net,Server, true_gradient, batch_recovered_middle_input)
            tf.Graph().finalize()
            # optimize data & label
            batch_dummy_data = optimizer_cafe.apply_gradients_data(
                cafe_iter, random_lists, cafe_gradient_x, batch_dummy_data)
            batch_dummy_label = optimizer_cafe.apply_gradients_label(
                cafe_iter, random_lists, cafe_gradient_y, batch_dummy_label)
            # assign dummy data
            dummy_data = assign_data(number_of_workers, batch_size, dummy_data, batch_dummy_data, random_lists)
            dummy_labels = assign_label(batch_size, dummy_labels, batch_dummy_label, random_lists)
            psnr = PSNR(batch_real_data, batch_dummy_data)
            # write down results
            if cafe_iter % 100 == 0:
                record(filename, [D, psnr, cafe_iter])
            # learning rate decay
            if cafe_iter % iter_decay == iter_decay -1 :
                cafe_learning_rate = cafe_learning_rate * decay_ratio
                # change the learning rate in the optimizer
                optimizer_cafe.lr = cafe_learning_rate
            # print results
            print(D, cafe_iter, cafe_learning_rate)
    # save recovered data as images
    visual_data(real_data, True)
    visual_data(dummy_data, False)
    # save recovered data & labels as numpy
    save_data(dummy_data, False)
    save_data(dummy_labels, True)
    print('Done')

if __name__ == "__main__":
    vfl_cafe()


