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
Losses class.

"""
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np


def l1_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.reduce_mean(tf.reshape(tf.abs(x-y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(tf.abs(x-y))
    return loss


def l2_loss(x, y, mask=None):
    xshape = x.shape
    if mask is None:
        loss = tf.reduce_mean(tf.reshape(tf.square(x-y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(tf.square(x-y))
    return loss


def leaf_l1_loss(xlist, y, masklist):
    loss_list = []
    xshape = xlist[0].shape
    for x, mask in zip(xlist, masklist):
        loss = tf.reduce_mean(tf.reshape(tf.abs(x-y), [xshape[0], -1]), axis=1)
        # tag of spoof
        tag = tf.reduce_sum(mask[:, 0])
        tag = tag / (tag + 1e-8)
        # spoof
        spoof_loss = tf.reduce_sum(loss * mask[:, 0]) / (tf.reduce_sum(mask[:, 0]) + 1e-8)
        # live
        live_loss = tf.reduce_sum(loss * mask[:, 1]) / (tf.reduce_sum(mask[:, 1]) + 1e-8)
        total_loss = (spoof_loss + live_loss)/2
        loss_list.append(total_loss*tag)
    loss = tf.reduce_mean(loss_list)
    return loss


def leaf_l2_loss(xlist, y, masklist):
    loss_list = []
    for x, mask in zip(xlist, masklist):
        xshape = x.shape
        print(x.shape, y.shape, mask.shape)
        input()
        # spoof
        spoof_loss = tf.reduce_mean(tf.reshape(tf.square(x-y), [xshape[0], -1]), axis=1)
        spoof_loss = tf.reduce_sum(loss * mask[:, 0]) / (tf.reduce_sum(mask[:, 0]) + 1e-8)
        # live
        live_loss = tf.reduce_mean(tf.reshape(tf.square(x - y), [xshape[0], -1]), axis=1)
        live_loss = tf.reduce_sum(loss * mask[:, 1]) / (tf.reduce_sum(mask[:, 1]) + 1e-8)
        loss = spoof_loss + live_loss
        loss_list.append(loss)

    return loss