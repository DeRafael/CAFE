#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
in this file we put all the parameters here
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' # device setting
max_iters = 20000
learning_rate_first_shot = 5e-3
learning_rate_double_shot = 1e-2
cafe_learning_rate = 0.01
max_cafe_iters = 5
learning_rate_fl = 1e-6
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7
filename = 'VFL_cafe_cifar10_'+str(cafe_learning_rate)
number_of_workers = 4
data_number = 800 # 40
test_data_number = 100
# img_size = 32
batch_size = 40 # 8
iter_decay = 2300 # how many iterations the learning rate decay once
iter_warm_up = 8000
decay_ratio = 1 # learning rate decay ratio

# central_file_name = 'central_process'


