# -*- coding: utf-8 -*-
# in this version we add other functional utilities
import copy as cp
import matplotlib.pyplot as plt
import math as m
import numpy as np
import tensorflow as tf
import random as r
import gc

# define cross entropy loss function
def compute_loss(true, pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(true, pred), axis=-1)

def compute_accuracy(true, pred):
    return tf.reduce_mean(tf.keras.metrics.categorical_accuracy(true, pred))

def dummy_data_init(number_of_workers, data_number, pretrain = False, true_label = None):
    '''
    In this function we initialize dummy data
    :param number_of_workers:
    :param data_number:
    :return: dummy_images, dummy_labels
    '''
    if pretrain:
        dummy_images = []
        for worker_index in range(number_of_workers):
            temp_dummy_image = np.load('result/' + str(worker_index) + '_dummy.npy')
            temp_dummy_image = tf.Variable(tf.convert_to_tensor(temp_dummy_image))
            dummy_images.append(temp_dummy_image)
        dummy_labels = np.load('result/labels_.npy')
        dummy_labels = tf.Variable(tf.convert_to_tensor(dummy_labels))
        return dummy_images, dummy_labels

    else:
        dummy_images = []
        for worker_index in range(number_of_workers):
            temp_dummy_image = tf.random.uniform(shape=[data_number, 14, 14], seed= worker_index + 1)
            # temp_dummy_image = tf.random.normal(shape=[data_number, 16, 16, 3], seed= n + 1)
            # temp_dummy_image = tf.zeros([data_number, 16, 16, 3])
            # temp_dummy_image = tf.ones([data_number, 16, 16, 3])
            temp_dummy_image = tf.Variable(temp_dummy_image)
            dummy_images.append(temp_dummy_image)
        if true_label == None:
            dummy_labels = tf.random.uniform(shape = [data_number, 10], seed = 0)
            # dummy_labels = tf.random.normal(shape=[data_number, 5], seed= 0)
        else:
            dummy_labels =  true_label
        dummy_labels = tf.Variable(dummy_labels)
        return dummy_images, dummy_labels

def dummy_middle_output_gradient_init(number_of_workers, data_number, feature_space):
    '''
    In this function we initialize middle output gradient
    :param number_of_workers:
    :param data_number:
    :return: feature space
    '''
    dummy_middle_output_gradient = []
    for worker_index in range(number_of_workers):
        temp_dummy_middle_output_gradient = tf.random.uniform(shape=[data_number, feature_space], minval=-8e-4,
                                                              maxval=8e-4, seed=worker_index + 1)
        temp_dummy_middle_output_gradient = tf.Variable(temp_dummy_middle_output_gradient)
        dummy_middle_output_gradient.append(temp_dummy_middle_output_gradient)
    return dummy_middle_output_gradient

def dummy_middle_input_init(number_of_workers, data_number, feature_space):
    '''
    In this function we initialize dummy data
    :param number_of_workers:
    :param data_number:
    :return: feature space
    '''
    dummy_middle_input = []
    for worker_index in range(number_of_workers):
        temp_dummy_middle_input = tf.random.uniform(shape=[data_number, feature_space], minval=0, maxval=8e-2,
                                                    seed=worker_index + 1)
        temp_dummy_middle_input = tf.Variable(temp_dummy_middle_input)
        dummy_middle_input.append(temp_dummy_middle_input)
    return dummy_middle_input

def list_real_data(number_of_workers, train_datasets, data_number):
    '''
    In this function we list all real data and put them in a big list
    :param number_of_workers:
    :param train_datasets:
    :return: real_images, real_labels
    '''
    real_labels = list(zip(*train_datasets))[-1][0]
    total_real_data = len(real_labels)
    r.seed(0)
    real_sample_list = r.sample(list(range(total_real_data)), data_number)
    real_labels = tf.gather(real_labels, real_sample_list)
    # real_labels = tf.reshape(tf.one_hot(real_labels, 5), (-1, 5))
    real_images = []
    for worker_index in range(number_of_workers):
        temp_images = list(zip(*train_datasets))[worker_index]
        real_images.append(tf.gather(temp_images[0], real_sample_list, axis = 0))
    return  real_images, real_labels

def take_gradient(number_of_workers, random_lists, real_data, real_labels, local_net, server):
    '''
    compute the real gradient
    :param number_of_workers:
    :param data_number:
    :param batchsize:
    :param real_images:
    :param real_labels:
    :param net:
    :return: true gradient
    '''
    true_gradient = []
    local_output = []
    middle_input = []
    middle_output = []
    batch_real_data = []
    real_tv_norm = []
    with tf.GradientTape(persistent = True) as tape:
        label = tf.gather(real_labels, random_lists, axis = 0)
        for worker_index in range(number_of_workers):
            # gradient tape
            # take the batch
            temp_data = tf.gather(real_data[worker_index], random_lists, axis=0)
            # compute output and loss
            temp_middle_input, temp_local_output, temp_middle_output = local_net[worker_index].forward(temp_data)
            # collect erms
            middle_input.append(temp_middle_input)
            middle_output.append(temp_middle_output)
            local_output.append(temp_local_output)
            batch_real_data.append(temp_data)

            # compute real TV norm
            temp_data = tf.reshape(temp_data, [-1, 14, 14,1])
            temp_tv_norm = tf.image.total_variation(temp_data)
            temp_tv_norm = tf.reduce_mean(temp_tv_norm, axis = 0)
            real_tv_norm.append(temp_tv_norm)
        # concatenate
        real_middle_input = tf.concat(middle_input, axis=1) # batch size x 2048
        real_local_output = tf.concat(local_output, axis=1) # batch size x 40
        # server part
        predict = server.forward(real_local_output)
        # compute loss
        loss = compute_loss(predict, label)
        # training accuracy
        train_acc = compute_accuracy(label, predict)

    # server gradient
    temp_server_true_gradient = tape.gradient(loss, server.trainable_variables)
    true_gradient.append(temp_server_true_gradient)
    # local gradients
    middle_output_gradient = []
    for worker_index in range(number_of_workers):
        temp_local_true_gradient = tape.gradient(loss, local_net[worker_index].trainable_variables)
        temp_middle_output_gradient = tape.gradient(loss, middle_output[worker_index])
        true_gradient.append(temp_local_true_gradient)
        middle_output_gradient.append(temp_middle_output_gradient)

    # compute aggregated TV norm
    real_tv_norm_aggregated = real_tv_norm[0]
    for worker_index in range(1, number_of_workers):
        real_tv_norm_aggregated += real_tv_norm[worker_index]
    real_tv_norm_aggregated = real_tv_norm_aggregated / number_of_workers
    print('real TV norm', real_tv_norm_aggregated.numpy(), end = '\t')
    return true_gradient, batch_real_data, middle_input, middle_output_gradient, loss, train_acc

def select_index(iter, data_number, batchsize):
    '''
    generate the batch index
    :param iter:
    :param number_of_workers:
    :param data_number:
    :param batchsize: batch size
    :return: random_lists
    '''
    r.seed(iter)
    random_lists = r.sample(list(range(data_number)), batchsize)
    return random_lists

def aggregate(gradients, number_of_workers):
    """
    Aggregate the gradients list
    :param gradients: the gradients list
    :param number_of_workers:
    :return: aggregated gradient
    """
    aggregated_gradient = []
    for l in range(len(gradients[0])):
        shape = gradients[0][l].numpy().shape
        temp_gradient = tf.Variable(tf.zeros(shape))
        for worker_index in range(number_of_workers):
            temp_gradient = temp_gradient + gradients[worker_index][l]
        # temp_gradient = temp_gradient / number_of_workers
        aggregated_gradient.append(temp_gradient)
    return aggregated_gradient

def take_batch_data(number_of_workers, dummy_images, dummy_labels, random_lists):
    '''
    Take batch:
    :param number_of_workers:
    :param dummy_images:
    :param dummy_labels:
    :param random_lists:
    :return: batch_dummy_data, batch_dummy_label
    '''
    batch_dummy_image = []
    # take the responding batch data
    for worker_index in range(number_of_workers):
        temp_dummy_image = tf.gather(dummy_images[worker_index], random_lists, axis=0)
        temp_dummy_image = tf.Variable(temp_dummy_image)
        batch_dummy_image.append(temp_dummy_image)
    temp_dummy_label = tf.gather(dummy_labels, random_lists, axis=0)
    batch_dummy_label = tf.Variable(temp_dummy_label)
    return batch_dummy_image, batch_dummy_label

def take_batch(number_of_workers, dummy_item, random_lists):
    '''
    Take batch:
    :param number_of_workers:
    :param dummy_images:
    :param dummy_labels:
    :param random_lists:
    :return: batch_dummy_data, batch_dummy_label
    '''
    batch_dummy_item = []
    # take the responding batch data
    for worker_index in range(number_of_workers):
        temp_dummy_item = tf.gather(dummy_item[worker_index], random_lists, axis=0)
        temp_dummy_item = tf.Variable(temp_dummy_item)
        batch_dummy_item.append(temp_dummy_item)
    return batch_dummy_item

def cafe(number_of_workers, batch_dummy_image, batch_dummy_label, local_net, server, real_gradient, real_middle_input):
    '''
    Core part of the algorithm: DLG
    :param number_of_workers:
    :param batch_dummy_image:
    :param batch_dummy_label:
    :param local_net:
    :param server
    :param real_gradient:
    :return: D, dlg_gradient_x, dlg_gradient_y
    '''
    # compute fake gradient
    with tf.GradientTape(persistent=True) as t:
        t.reset()
        # go through all the workers
        fake_gradient = []
        fake_local_output = []
        fake_middle_input = []
        for worker_index in range(number_of_workers):
            t.watch(batch_dummy_image[worker_index])
            # input images
            temp_middle_input, temp_local_output, temp_middle_output = local_net[worker_index].forward(
                batch_dummy_image[worker_index])
            fake_local_output.append(temp_local_output)
            fake_middle_input.append(temp_middle_input)
        del temp_local_output, temp_middle_input
        gc.collect()
        # concat
        dummy_middle_input = tf.concat(fake_middle_input, axis = 1)
        dummy_local_output = tf.concat(fake_local_output, axis = 1)
        # dummy_middle_input = tf.reduce_mean(dummy_middle_input, axis = 2)

        # server part
        predict = server.forward(dummy_local_output)
        # compute loss
        t.watch(batch_dummy_label)
        true = tf.nn.softmax(batch_dummy_label)
        loss = compute_loss(true, predict)

        # compute fake gradient
        temp_server_true_gradient = t.gradient(loss, server.trainable_variables)
        fake_gradient.append(temp_server_true_gradient)
        for worker_index in range(number_of_workers):
            temp_local_fake_gradient = t.gradient(loss, local_net[worker_index].trainable_variables)
            fake_gradient.append(temp_local_fake_gradient)
        del temp_server_true_gradient
        del temp_local_fake_gradient
        gc.collect()

        # compute D loss
        D = 0
        for layer in range(len(real_gradient)):
            for gr, gf in zip(real_gradient[layer], fake_gradient[layer]):
                gr = tf.reshape(gr, [-1, 1])
                gf = tf.reshape(gf, [-1, 1])
                # D_norm = tf.norm(gr - gf) ** 2
                # sigma = tf.math.reduce_std(gr) ** 2
                D += tf.norm(gr - gf) ** 2
                # D += 1 - tf.math.exp(- D_norm / sigma)
        D *= 100

        # compute local output norm
        D_local_output_norm = 0
        for r_real_middle_input, dummy_middle_input in zip(real_middle_input, dummy_middle_input):
            temp_input_norm = tf.norm(r_real_middle_input - dummy_middle_input) ** 2
            D_local_output_norm += temp_input_norm
        del temp_input_norm
        gc.collect()

        print("CAFE loss: %.5f" % D.numpy(), end = '\t')
        print('Input norm:', D_local_output_norm.numpy(), end = '\t')

        # compute tv norm
        tv_norm = []
        for worker_index in range(number_of_workers):
            temp_data = batch_dummy_image[worker_index]
            temp_data = tf.reshape(temp_data, [-1, 14, 14, 1])
            temp_tv_norm = tf.image.total_variation(temp_data)
            temp_tv_norm = tf.reduce_mean(temp_tv_norm, axis = 0)
            tv_norm.append(temp_tv_norm)
        del temp_tv_norm
        gc.collect()

        # compute aggregated TV norm
        tv_norm_aggregated = tv_norm[0]
        for worker_index in range(1, number_of_workers):
            tv_norm_aggregated += tv_norm[worker_index]
        tv_norm_aggregated = tv_norm_aggregated / number_of_workers
        tv_norm_aggregated = tf.reduce_mean(tv_norm_aggregated)
        # D += tv_norm_aggregated
        print('with Tv norm', tv_norm_aggregated.numpy(), end = '\t')

        '''
        compute cafe gradient
        '''
        cafe_gradient_x = []
        cafe_gradient_y = t.gradient(D, batch_dummy_label) # label known
        for worker_index in range(number_of_workers):
            temp_tv_norm_gradient = t.gradient(tv_norm[worker_index], batch_dummy_image[worker_index])
            temp_local_output_gradient = t.gradient(D_local_output_norm, batch_dummy_image[worker_index])
            temp_dlg_gradient = 1e-4 * t.gradient(D, batch_dummy_image[worker_index])
            temp_cafe_gradient_x = 1e-3 * temp_local_output_gradient + temp_dlg_gradient
            # add Tv norm gradient
            if tv_norm_aggregated.numpy() > 25:
                temp_cafe_gradient_x = temp_cafe_gradient_x + 1e-4 * temp_tv_norm_gradient
            cafe_gradient_x.append(temp_cafe_gradient_x)
        return D.numpy(), cafe_gradient_x, cafe_gradient_y

def assign_to_dummy(number_of_workers, batchsize, dummy_item, batch_dummy_item, random_lists):
    for batch_index in range(batchsize):
        for worker_index in range(number_of_workers):
            dummy_item[worker_index][random_lists[batch_index], :].assign(
                batch_dummy_item[worker_index][batch_index, :])
    return dummy_item

def assign_data(number_of_workers, batchsize, dummy_item, batch_dummy_item, random_lists):
    for batch_index in range(batchsize):
        for worker_index in range(number_of_workers):
            dummy_item[worker_index][random_lists[batch_index], :, :].assign(
                batch_dummy_item[worker_index][batch_index, :, :])
    return dummy_item

def assign_label(batchsize, dummy_labels, batch_dummy_label, random_lists):
    for batch_index in range(batchsize):
        dummy_labels[random_lists[batch_index], :].assign(batch_dummy_label[batch_index, :])
    return dummy_labels

def record(filename, record_list):
    '''
    Write parameters into the txt file
    :param filename:
    :param record_list: record list
    :return:
    '''
    file = open(filename + '.txt', 'a+')
    for i in range(len(record_list)):
        file.write(str(record_list[i]))
        if i == len(record_list) - 1:
            file.write('\n')
        else:
            file.write('\t')
    file.close()

class Optimizer_for_middle_input():
    '''
    Optimizer for middle input
    '''
    def __init__(self, number_of_workers, data_number, learning_rate, feature_space=2048, beta1=0.9, beta2=0.999,
                 epsilon=1e-7):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # initialize m and v (momentum)
        self.h_data = []
        self.v_data = []
        self.number_of_workers = number_of_workers
        for worker_index in range(number_of_workers):
            self.h_data.append(tf.Variable(tf.zeros([data_number, feature_space])))
            self.v_data.append(tf.Variable(tf.zeros([data_number, feature_space])))

    def apply_gradients(self, iter, batchsize, random_lists, gradient, theta):
        theta_new = []
        # optimize data
        # learning rate decay
        temp_lr = self.lr * m.sqrt(1 - self.beta2 ** (iter + 1)) / (1 - self.beta1 ** (iter + 1))
        for worker_index in range(self.number_of_workers):
            # take out the h
            h = tf.gather(self.h_data[worker_index], random_lists, axis=0)
            # update h
            h = self.beta1 * h + (1 - self.beta1) * gradient[worker_index]
            # take out the v
            v = tf.gather(self.v_data[worker_index], random_lists, axis=0)
            # update v
            v = self.beta2 * v + (1 - self.beta2) * tf.math.square(gradient[worker_index])
            # update dummy data
            h_hat = h / (1 - self.beta1 ** (iter+1))
            v_hat = v / (1 - self.beta2 ** (iter+1))
            for batch_index in range(batchsize):
                self.h_data[worker_index][random_lists[batch_index], :].assign(h[batch_index, :])
                self.v_data[worker_index][random_lists[batch_index], :].assign(v[batch_index, :])
            temp_theta = theta[worker_index] - temp_lr * h_hat / (tf.math.sqrt(v_hat) + self.epsilon)
            theta_new.append(temp_theta)
        return theta_new

class Optimizer_for_cafe():
    '''
    Adam optimizer
    '''
    def __init__(self, number_of_workers, data_number, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # initialize m and v (momentum)
        self.h_data = []
        self.v_data = []
        self.number_of_workers = number_of_workers

        for worker_index in range(number_of_workers):
            self.h_data.append(tf.Variable(tf.zeros([data_number, 14, 14])))
            self.v_data.append(tf.Variable(tf.zeros([data_number, 14, 14])))
        self.h_label = tf.Variable(tf.zeros([data_number, 10]))
        self.v_label = tf.Variable(tf.zeros([data_number, 10]))


    def apply_gradients_data(self, iter, random_lists, gradient, theta):
        '''
        In this function, we optimize theta
        :param iter:
        :param random_lists:
        :param gradient: a list
        :param theta: a list
        :param data: optimize data or label
        :return: theta_new
        '''
        # update m
        theta_new = []
        # optimize data
        # learning rate decay
        temp_lr = self.lr * m.sqrt(1 - self.beta2 ** (iter + 1)) / (1 - self.beta1 ** (iter + 1))
        for worker_index in range(self.number_of_workers):
            # take out the h
            h = tf.gather(self.h_data[worker_index], random_lists, axis=0)
            # update h
            h = self.beta1 * h + (1 - self.beta1) * gradient[worker_index]
            # h = (1 - self.beta1) * gradient[worker_index]
            # take out the v
            v = tf.gather(self.v_data[worker_index], random_lists, axis=0)
            # update v
            v = self.beta2 * v + (1 - self.beta2) * tf.math.square(gradient[worker_index])
            # v = (1 - self.beta2) * tf.math.square(gradient[worker_index])
            # compute h_hat, v_hat
            h_hat = h / (1 - self.beta1 ** (iter+1))
            v_hat = v / (1 - self.beta2 ** (iter+1))
            temp_theta = theta[worker_index] - temp_lr * h_hat / (tf.math.sqrt(v_hat) + self.epsilon)
            theta_new.append(temp_theta)
            # store h and v

            for batch_index in range(len(random_lists)):
                self.h_data[worker_index][random_lists[batch_index], :, :].assign(h[batch_index, :, :])
                self.v_data[worker_index][random_lists[batch_index], :, :].assign(v[batch_index, :, :])

        return theta_new

    def apply_gradients_label(self, iter, random_lists, gradient, theta):
        # learning rate decay
        temp_lr = self.lr * m.sqrt(1 - self.beta2 ** (iter + 1)) / (1 - self.beta1 ** (iter + 1))
        # take out the h
        h = tf.gather(self.h_label, random_lists, axis=0)
        # update h
        h = self.beta1 * h + (1 - self.beta1) * gradient
        # h = (1 - self.beta1) * gradient
        # take out the v
        v = tf.gather(self.v_label, random_lists, axis=0)
        # update v
        v = self.beta2 * v + (1 - self.beta2) * tf.math.square(gradient)
        # v = (1 - self.beta2) * tf.math.square(gradient)
        # compute h_hat, v_hat
        h_hat = h / (1 - self.beta1 ** (iter+1))
        v_hat = v / (1 - self.beta2 ** (iter+1))
        # update dummy data
        theta_new = theta - temp_lr * h_hat / (tf.math.sqrt(v_hat) + self.epsilon)
        # store h and v

        for batch_index in range(len(random_lists)):
            self.h_label[random_lists[batch_index], :].assign(h[batch_index, :])
            self.v_label[random_lists[batch_index], :].assign(v[batch_index, :])

        return theta_new

def visual_data(data, real):
    '''
    In this function we visualize the data
    :param data: data to be visualized (list)
    :real True or false
    :return:
    '''
    number_of_worker = len(data)
    if real:
        # save real iamge
        for worker_index in range(number_of_worker):
            data_number = data[worker_index].numpy().shape[0]
            for data_index in range(data_number):
                data_to_be_visualized = data[worker_index][data_index, :, :].numpy()
                data_to_be_visualized = tf.reshape(data_to_be_visualized, [14, 14])
                plt.imshow(data_to_be_visualized)
                plt.savefig('result/' + str(worker_index) + '/' + str(data_index) + 'real.png')
                plt.close()
    else:
        # save real iamge
        for worker_index in range(number_of_worker):
            data_number = data[worker_index].numpy().shape[0]
            for data_index in range(data_number):
                data_to_be_visualized = data[worker_index][data_index, :, :].numpy()
                data_to_be_visualized = tf.reshape(data_to_be_visualized, [14, 14])
                plt.imshow(data_to_be_visualized)
                plt.savefig('result/' + str(worker_index) + '/' + str(data_index) + 'dummy.png')
                plt.close()

def PSNR(batch_real_image, batch_dummy_image):
    '''
    compute PSNR
    :param batch_real_image:
    :param batch_dummy_image:
    :return:
    '''
    psnr = []
    for worker_index in range(len(batch_real_image)):
        dummy = tf.reshape(tf.clip_by_value(batch_dummy_image[worker_index], 0, 1), [-1, 14, 14, 1])
        real = tf.reshape(batch_real_image[worker_index], [-1, 14, 14, 1])
        psnr.append(tf.reduce_mean(tf.image.psnr(real, dummy, 1.0)))
    aggregated_psnr = tf.reduce_mean(psnr)
    print('psnr value:', aggregated_psnr.numpy(), end='\t')
    return aggregated_psnr.numpy()

def save_data(data, labels):
    '''
    In this function we save the data into npy format
    :param data: dummy(real) data
    :param real: True or False
    :return:
    '''
    if labels:
        # save labels
        data_to_be_save = data.numpy()
        np.save('result/labels_.npy', data_to_be_save)
    else:
        number_of_workers = len(data)
        # save dummy data
        for worker_index in range(number_of_workers):
            data_to_be_save = data[worker_index].numpy()
            np.save('result/' + str(worker_index) + '_dummy.npy', data_to_be_save)

def test(number_of_workers, test_data, test_labels, local_net, server):
    '''
    compute the real gradient
    :param number_of_workers:
    :param data_number:
    :param batchsize:
    :param real_images:
    :param real_labels:
    :param net:
    :return: true gradient
    '''
    local_output = []
    for worker_index in range(number_of_workers):
        # compute output
        temp_middle_input, temp_local_output, temp_middle_output = local_net[worker_index].forward(
            test_data[worker_index])
        # collect terms
        local_output.append(temp_local_output)
    # concatenate
    real_local_output = tf.concat(local_output, axis=1) # batch size x 40
    # server part
    predict = server.forward(real_local_output)
    # compute loss
    loss = compute_loss(test_labels, predict)
    # training accuracy
    test_acc = compute_accuracy(test_labels, predict)

    return loss, test_acc






