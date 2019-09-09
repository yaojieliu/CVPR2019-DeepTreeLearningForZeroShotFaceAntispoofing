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
DTN Model class.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from model.utils import CRU, TRU, SFL, Conv, Downsample, Error
from model.loss import l1_loss, l2_loss, leaf_l1_loss, leaf_l2_loss

############################################################
#  Deep Tree Network
############################################################
class DTN(tf.keras.models.Model):
    def __init__(self, filters, config):
        super(DTN, self).__init__()
        self.config = config
        layer = [1, 2, 4, 8, 16]
        self.conv1 = Conv(filters, 5, apply_batchnorm=False)
        # CRU
        self.cru0 = CRU(filters)
        self.cru1 = CRU(filters)
        self.cru2 = CRU(filters)
        self.cru3 = CRU(filters)
        self.cru4 = CRU(filters)
        self.cru5 = CRU(filters)
        self.cru6 = CRU(filters)
        self.cru6 = CRU(filters)
        # TRU
        alpha = config.TRU_PARAMETERS['alpha']
        beta = config.TRU_PARAMETERS['beta']
        self.tru0 = TRU(filters, '1', alpha, beta)
        self.tru1 = TRU(filters, '2', alpha, beta)
        self.tru2 = TRU(filters, '3', alpha, beta)
        self.tru3 = TRU(filters, '4', alpha, beta)
        self.tru4 = TRU(filters, '5', alpha, beta)
        self.tru5 = TRU(filters, '6', alpha, beta)
        self.tru6 = TRU(filters, '7', alpha, beta)
        # SFL
        self.sfl0 = SFL(filters)
        self.sfl1 = SFL(filters)
        self.sfl2 = SFL(filters)
        self.sfl3 = SFL(filters)
        self.sfl4 = SFL(filters)
        self.sfl5 = SFL(filters)
        self.sfl6 = SFL(filters)
        self.sfl7 = SFL(filters)

    @tf.function
    def call(self, x, label, training):
        mask_spoof = label
        mask_live = 1 - label
        ''' Tree Level 1 '''
        x = self.conv1(x, training)
        x_cru0 = self.cru0(x)
        x_tru0, route_value0, tru0_loss = self.tru0(x_cru0, mask_spoof, True)

        ''' Tree Level 2 '''
        x_cru00 = self.cru1(x_cru0, True)
        x_cru01 = self.cru2(x_cru0, True)
        x_tru00, route_value00, tru00_loss = self.tru1(x_cru00, x_tru0[0], True)
        x_tru01, route_value01, tru01_loss = self.tru2(x_cru01, x_tru0[1], True)

        ''' Tree Level 3 '''
        x_cru000 = self.cru3(x_cru00, True)
        x_cru001 = self.cru4(x_cru00, True)
        x_cru010 = self.cru5(x_cru01, True)
        x_cru011 = self.cru6(x_cru01, True)
        x_tru000, route_value000, tru000_loss = self.tru3(x_cru00, x_tru00[0], True)
        x_tru001, route_value001, tru001_loss = self.tru4(x_cru01, x_tru00[1], True)
        x_tru010, route_value010, tru010_loss = self.tru5(x_cru00, x_tru01[0], True)
        x_tru011, route_value011, tru011_loss = self.tru6(x_cru01, x_tru01[1], True)

        ''' Tree Level 4 '''
        map0, cls0 = self.sfl0(x_cru000, True)
        map1, cls1 = self.sfl1(x_cru000, True)
        map2, cls2 = self.sfl2(x_cru001, True)
        map3, cls3 = self.sfl3(x_cru001, True)
        map4, cls4 = self.sfl4(x_cru010, True)
        map5, cls5 = self.sfl5(x_cru010, True)
        map6, cls6 = self.sfl6(x_cru011, True)
        map7, cls7 = self.sfl7(x_cru011, True)
        x_tru0000 = tf.concat([x_tru000[0], mask_live], axis=1)
        x_tru0001 = tf.concat([x_tru000[1], mask_live], axis=1)
        x_tru0010 = tf.concat([x_tru001[0], mask_live], axis=1)
        x_tru0011 = tf.concat([x_tru001[1], mask_live], axis=1)
        x_tru0100 = tf.concat([x_tru010[0], mask_live], axis=1)
        x_tru0101 = tf.concat([x_tru010[1], mask_live], axis=1)
        x_tru0110 = tf.concat([x_tru011[0], mask_live], axis=1)
        x_tru0111 = tf.concat([x_tru011[1], mask_live], axis=1)
        ''' Output '''
        maps = [map0, map1, map2, map3, map4, map5, map6, map7]
        clss = [cls0, cls1, cls2, cls3, cls4, cls5, cls6, cls7]
        route_value = [route_value0, route_value00, route_value01,
                       route_value000, route_value001, route_value010, route_value011]
        leaf_node_mask = [x_tru0000, x_tru0001, x_tru0010, x_tru0011, x_tru0100, x_tru0101, x_tru0110, x_tru0111]

        # for the training
        if training:
            route_loss = [tru0_loss[0], tru00_loss[0], tru01_loss[0],
                          tru000_loss[0], tru001_loss[0], tru010_loss[0], tru011_loss[0]]
            recon_loss = [tru0_loss[1], tru00_loss[1], tru01_loss[1],
                          tru000_loss[1], tru001_loss[1], tru010_loss[1], tru011_loss[1]]
            mu_update = [self.tru0.project.mu_of_visit+0,
                         self.tru1.project.mu_of_visit+0,
                         self.tru2.project.mu_of_visit+0,
                         self.tru3.project.mu_of_visit+0,
                         self.tru4.project.mu_of_visit+0,
                         self.tru5.project.mu_of_visit+0,
                         self.tru6.project.mu_of_visit+0]

            return maps, clss, route_value, leaf_node_mask, [route_loss, recon_loss], mu_update
        else:
            return maps, clss, route_value, leaf_node_mask


class Model:
    def __init__(self, config):
        self.config = config
        # model
        self.dtn = DTN(32, config)
        # model optimizer
        self.dtn_op = tf.compat.v1.train.AdamOptimizer(config.LEARNING_RATE, beta1=0.5)
        # model losses
        self.depth_map_loss = Error()
        self.class_loss = Error()
        self.route_loss = Error()
        self.recon_loss = Error()
        # model saving setting
        self.last_epoch = 0
        self.checkpoint_manager = []

    def compile(self):
        checkpoint_dir = self.config.LOG_DIR
        checkpoint = tf.train.Checkpoint(dtn=self.dtn,
                                         dtn_optimizer=self.dtn_op)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)
        last_checkpoint = self.checkpoint_manager.latest_checkpoint
        checkpoint.restore(last_checkpoint)
        if last_checkpoint:
            self.last_epoch = int(last_checkpoint.split('-')[-1])
            print("Restored from {}".format(last_checkpoint))
        else:
            print("Initializing from scratch.")

    def train(self, train, val=None):
        config = self.config
        step_per_epoch = config.STEPS_PER_EPOCH
        step_per_epoch_val = config.STEPS_PER_EPOCH_VAL
        epochs = config.MAX_EPOCH

        # data stream
        it = train.feed
        global_step = 0
        if val is not None:
            it_val = val.feed
        for epoch in range(self.last_epoch, epochs):
            start = time.time()
            # define the
            self.dtn_op = tf.compat.v1.train.AdamOptimizer(config.LEARNING_RATE, beta1=0.5)
            ''' train phase'''
            for step in range(step_per_epoch):
                depth_map_loss, class_loss, route_loss, recon_loss = self.train_one_step(next(it), global_step, True)
                # display loss
                global_step += 1
                print('Epoch {:d}-{:d}/{:d}: Map:{:3g}, Cls:{:3g}, Route:{:3g}, Recon:{:3g}   '.format(
                                                            epoch + 1, step + 1, step_per_epoch,
                                                            self.depth_map_loss(depth_map_loss),
                                                            self.class_loss(class_loss),
                                                            self.route_loss(route_loss),
                                                            self.recon_loss(recon_loss)), end='\r')

            # save the model
            if (epoch + 1) % 1 == 0:
                self.checkpoint_manager.save(checkpoint_number=epoch + 1)
            print('\n')

            ''' eval phase'''
            if val is not None:
                for step in range(step_per_epoch_val):
                    depth_map_loss, class_loss, route_loss, recon_loss = self.train_one_step(next(it_val), global_step, False)
                    # display something
                    print('Epoch {:d}-Val-{:d}/{:d}: Map:{:3g}, Cls:{:3g}, Route:{:3g}, Recon:{:3g}   '.format(
                                                            epoch + 1, step+1, step_per_epoch_val,
                                                            self.depth_map_loss(depth_map_loss, val=1),
                                                            self.class_loss(class_loss, val=1),
                                                            self.route_loss(route_loss, val=1),
                                                            self.recon_loss(recon_loss, val=1)), end='\r')
                self.depth_map_loss.reset_val()
                self.class_loss.reset_val()
                self.route_loss.reset_val()
                self.recon_loss.reset_val()

            # time of one epoch
            print('Time taken for epoch {} is {:3g} sec'.format(epoch + 1, time.time() - start))
        return 0

    def train_one_step(self, data_batch, step, training):
        dtn = self.dtn
        dtn_op = self.dtn_op
        image, dmap, labels = data_batch
        with tf.GradientTape() as tape:
            dmap_pred, cls_pred, route_value, leaf_node_mask, tru_loss, mu_update = dtn(image, labels, training)

            # supervised feature loss
            depth_map_loss = leaf_l1_loss(dmap_pred, tf.image.resize(dmap, [32, 32]), leaf_node_mask)
            class_loss = leaf_l1_loss(cls_pred, labels, leaf_node_mask)
            supervised_loss = depth_map_loss + class_loss

            # unsupervised tree loss
            route_loss = tf.reduce_mean(tf.stack(tru_loss[0], axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            recon_loss = tf.reduce_mean(tf.stack(tru_loss[1], axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            unsupervised_loss = 0.01*route_loss + recon_loss

            # total loss
            loss = supervised_loss + unsupervised_loss

        # back-propagate
        gradients = tape.gradient(loss, dtn.variables)
        dtn_op.apply_gradients(zip(gradients, dtn.variables))

        # Update mean values for each tree node
        mu_update_rate = self.config.TRU_PARAMETERS["mu_update_rate"]
        mu = [dtn.tru0.project.mu, dtn.tru1.project.mu, dtn.tru2.project.mu, dtn.tru3.project.mu,
              dtn.tru4.project.mu, dtn.tru5.project.mu, dtn.tru6.project.mu]
        for mu, mu_of_visit in zip(mu,mu_update):
            if step == 0:
                update_mu = mu_of_visit
            else:
                update_mu = mu_of_visit * mu_update_rate + mu * (1 - mu_update_rate)
            K.set_value(mu, update_mu)

        return depth_map_loss, class_loss, route_loss, recon_loss

    def test(self, test):
        # TO DO
        dtn = self.dtn
        config = self.config
        step_per_epoch = config.STEPS_PER_EPOCH
        step_per_epoch_val = config.STEPS_PER_EPOCH_VAL
        epochs = config.MAX_EPOCH

        # data stream
        it = test.feed
        dtn = self.dtn
        image, dmap, labels = next(it)
        dmap_pred, cls_pred, route_value, leaf_node_mask, tru_loss, mu_update = dtn(image, labels, False)

        return 0