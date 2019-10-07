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
Data Loading class.

"""
import tensorflow as tf
import numpy as np
import glob


class Dataset():
    def __init__(self, config, mode):
        self.config = config
        if self.config.MODE == 'training':
            self.input_tensors = self.inputs_for_training(mode)
        else:
            self.input_tensors, self.name_list = self.inputs_for_testing()
        self.feed = iter(self.input_tensors)

    def inputs_for_training(self, mode):
        autotune = tf.data.experimental.AUTOTUNE
        if mode == 'train':
            data_dir = self.config.DATA_DIR
        else:
            data_dir = self.config.DATA_DIR_VAL
        data_samples = []
        for _dir in data_dir:
            _list = glob.glob(_dir+'/*.dat')
            data_samples += _list
        shuffle_buffer_size = len(data_samples)
        dataset = tf.data.Dataset.from_tensor_slices(data_samples)
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(-1)
        dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        return dataset

    def inputs_for_testing(self):
        autotune = tf.data.experimental.AUTOTUNE
        data_dir = self.config.DATA_DIR
        data_samples = []
        for _dir in data_dir:
            _list = sorted(glob.glob(_dir + '/*.dat'))
            data_samples += _list
        dataset = tf.data.Dataset.from_tensor_slices(data_samples)
        dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        return dataset, data_samples

    def parse_fn(self, file):
        config = self.config
        image_size = config.IMAGE_SIZE
        dmap_size = config.MAP_SIZE
        label_size = 1

        def _parse_function(_file):
            _file = _file.decode('UTF-8')
            image_bytes = image_size * image_size * 3
            dmap_bytes = dmap_size * dmap_size
            bin = np.fromfile(_file, dtype='uint8')
            image = np.transpose(bin[0:image_bytes].reshape((3, image_size, image_size)) / 255, (1, 2, 0))
            dmap  = np.transpose(bin[image_bytes:image_bytes+dmap_bytes].reshape((1, dmap_size, dmap_size)) / 255, (1, 2, 0))
            label = bin[image_bytes+dmap_bytes:image_bytes+dmap_bytes+label_size] / 1
            dmap1 = dmap * (1-label)
            dmap2 = np.ones_like(dmap) * label
            dmap = np.concatenate([dmap1, dmap2], axis=2)

            return image.astype(np.float32), dmap.astype(np.float32), label.astype(np.float32)

        image_ts, dmap_ts, label_ts = tf.numpy_function(_parse_function, [file], [tf.float32, tf.float32, tf.float32])
        image_ts = tf.ensure_shape(image_ts, [config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
        dmap_ts  = tf.ensure_shape(dmap_ts,  [config.MAP_SIZE, config.MAP_SIZE, 2])
        label_ts = tf.ensure_shape(label_ts, [1])
        return image_ts, dmap_ts, label_ts
