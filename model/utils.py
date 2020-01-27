# Copyright 2019
#
# Yaojie Liu, Joel Stehouwer, Amin Jourabloo, Xiaoming Liu, Michigan State University
#
# All Rights Reserved.
#
# This research is based upon work supported by the Office of the Director of
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and
# conclusions contained herein are those of the authors and should not be
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The
# U.S. Government is authorized to reproduce and distribute reprints for
# Governmental purposes not withstanding any copyright annotation thereon.
# ==============================================================================
"""
DTN for Zero-shot Face Anti-spoofing
DTN Basic Blocks class.

"""
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from model.loss import l1_loss, l2_loss


class Error:
    def __init__(self):
        self.value = 0
        self.value_val = 0
        self.step = 0
        self.step_val = 0

    def __call__(self, update, val=0):
        if val == 1:
            self.value_val += update
            self.step_val += 1
            return self.value_val / self.step_val
        else:
            self.value += update
            self.step += 1
            return self.value / self.step

    def reset(self):
        self.value = 0
        self.value_val = 0
        self.step = 0
        self.step_val = 0


class Linear(layers.Layer):
    def __init__(self, idx, alpha, beta, input_dim=32):
        super(Linear, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        initializer0 = tf.zeros_initializer()
        self.v = tf.Variable(initial_value=initializer(shape=(1, input_dim), dtype='float32'),
                             trainable=True, name='tru/v/'+idx)
        self.mu = tf.Variable(initial_value=initializer0(shape=(1, input_dim), dtype='float32'),
                              trainable=True, name='tru/mu/'+idx)
        # training hyper-parameters
        self.alpha = alpha
        self.beta = beta
        # mean, eigenvalue and trace for each mini-batch
        self.mu_of_visit = 0
        self.eigenvalue = 0.
        self.trace = 0.

    def call(self, x, mask, training):
        norm_v = self.v / (tf.norm(self.v) + 1e-8)
        norm_v_t = tf.transpose(norm_v, [1, 0])
        num_of_visit = tf.reduce_sum(mask)

        if training and num_of_visit > 1:
            # use only the visiting samples
            index = tf.where(tf.greater(mask[:, 0], tf.constant(0.)))
            index_not = tf.where(tf.equal(mask[:, 0], tf.constant(0.)))
            x_sub = tf.gather_nd(x, index) - tf.stop_gradient(self.mu)
            x_not = tf.gather_nd(x, index_not)
            x_sub_t = tf.transpose(x_sub, [1, 0])

            # compute the covariance matrix, eigenvalue, and the trace
            covar = tf.matmul(x_sub_t, x_sub) / num_of_visit
            eigenvalue = tf.reshape(tf.matmul(tf.matmul(norm_v, covar), norm_v_t), [])
            trace = tf.linalg.trace(covar)
            # compute the route loss
            # print(tf.exp(-self.alpha * eigenvalue), self.beta * trace)
            route_loss = tf.exp(-self.alpha * eigenvalue) + self.beta * trace
            uniq_loss = -tf.reduce_mean(tf.square(tf.matmul(x_sub, norm_v_t))) + \
                         tf.reduce_mean(tf.square(tf.matmul(x_not, norm_v_t)))
            # compute mean and response for this batch
            self.mu_of_visit = tf.reduce_mean(x_sub, axis=0, keepdims=True)
            self.eigenvalue = eigenvalue
            self.trace = trace
            x -= tf.stop_gradient(self.mu_of_visit)
            route_value = tf.matmul(x, norm_v_t)
        else:
            self.mu_of_visit = self.mu
            self.eigenvalue = 0.
            self.trace = 0.
            x -= self.mu
            route_value = tf.matmul(x, norm_v_t)
            route_loss = 0.
            uniq_loss = 0.

        return route_value, route_loss, uniq_loss


class Downsample(tf.keras.Model):
    def __init__(self, filters, size, padding='SAME', apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv1 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=2,
                                   padding=padding,
                                   kernel_initializer=initializer,
                                   use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x, training):
        x = self.up_conv(x)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Conv(tf.keras.Model):
    def __init__(self, filters, size, stride=1, activation=True, padding='SAME', apply_batchnorm=True):
        super(Conv, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.activation = activation
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv1 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=stride,
                                   padding=padding,
                                   kernel_initializer=initializer,
                                   use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        return x


class Dense(tf.keras.Model):
    def __init__(self, filters, activation=True, apply_batchnorm=True, apply_dropout=False):
        super(Dense, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.activation = activation
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.dense = layers.Dense(filters,
                                  kernel_initializer=initializer,
                                  use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, training):
        x = self.dense(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        return x


class CRU(tf.keras.Model):

    def __init__(self, filters, size=3, stride=2, apply_batchnorm=True):
        super(CRU, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.stride = stride
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)

        self.conv1 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)
        self.conv2 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)
        self.conv3 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)
        self.conv4 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)

        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        # first residual block
        _x = self.conv1(x)
        _x = self.batchnorm1(_x, training=training)
        _x = tf.nn.leaky_relu(_x)
        _x = self.conv2(_x)
        _x = self.batchnorm2(_x, training=training)
        _x  = x + _x
        x  = tf.nn.leaky_relu(_x)

        # second residual block
        _x = self.conv3(x)
        _x = self.batchnorm3(_x, training=training)
        _x = tf.nn.leaky_relu(_x)
        _x = self.conv4(_x)
        _x = self.batchnorm4(_x, training=training)
        _x = x + _x
        x = tf.nn.leaky_relu(_x)

        if self.stride > 1:
            x = tf.nn.max_pool(x, 3, 2, padding='SAME')
        return x


class TRU(tf.keras.Model):

    def __init__(self, filters, idx, alpha=1e-3, beta=1e-4, size=3, apply_batchnorm=True):
        super(TRU, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        # variables
        self.conv1 = Downsample(filters, size)
        self.conv2 = Downsample(filters, size)
        self.conv3 = Downsample(filters, size)
        self.flatten = layers.Flatten()
        self.project = Linear(idx, alpha, beta, input_dim=2048)


    def call(self, x, mask, training):
        # Downsampling
        x_small = self.conv1(x, training=training)
        depth = 0
        if x_small.shape[1] > 16:
            x_small = self.conv2(x_small, training=training)
            depth += 1
            if x_small.shape[1] > 16:
                x_small = self.conv3(x_small, training=training)
                depth += 1
        x_small_shape = x_small.shape
        x_flatten = self.flatten(tf.nn.avg_pool(x_small, ksize=3, strides=2, padding='SAME'))

        # PCA Projection
        route_value, route_loss, uniq_loss = self.project(x_flatten, mask, training=training)

        # Generate the splitting mask
        mask_l = mask * tf.cast(tf.greater_equal(route_value, tf.constant(0.)), tf.float32)
        mask_r = mask * tf.cast(tf.less(route_value, tf.constant(0.)), tf.float32)

        return [mask_l, mask_r], route_value, [route_loss, uniq_loss]


class SFL(tf.keras.Model):

    def __init__(self, filters, size=3, apply_batchnorm=True):
        super(SFL, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        # depth map
        self.cru1 = CRU(filters, size, stride=1)
        self.conv1 = Conv(2, size, activation=False, apply_batchnorm=False)

        # class
        self.conv2 = Downsample(filters*1, size)
        self.conv3 = Downsample(filters*1, size)
        self.conv4 = Downsample(filters*2, size)
        self.conv5 = Downsample(filters*4, 4, padding='VALID')
        self.flatten = layers.Flatten()
        self.fc1 = Dense(256)
        self.fc2 = Dense(1, activation=False, apply_batchnorm=False)

        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, training):
        # depth map branch
        xd = self.cru1(x)
        xd = self.conv1(xd)
        dmap = tf.nn.sigmoid(xd)
        # class branch
        x = self.conv2(x)  # 16*16*32
        x = self.conv3(x)  # 8*8*64
        x = self.conv4(x)  # 4*4*128
        x = self.conv5(x)  # 1*1*256
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        cls = self.fc2(x)
        return dmap, cls

