# Copyright 2018
# 
# Yaojie Liu, Amin Jourabloo, Xiaoming Liu, Michigan State University
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
Base Configurations class.

"""
import numpy as np
import tensorflow as tf

# Base Configuration Class
class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # GPU Usage
    GPU_USAGE = 1

    # Log and Model Storage Default
    LOG_DIR = './log/DTN'

    # Input Data Meta
    IMAGE_SIZE = 256
    MAP_SIZE = 64

    TRU_PARAMETERS = {
        "alpha": 1e-3,
        "beta": 1e-2,
        "mu_update_rate": 1e-3,
    }

    # Training Meta
    STEPS_PER_EPOCH = 1000
    MAX_EPOCH = 40
    NUM_EPOCHS_PER_DECAY = 12.0   # Epochs after which learning rate decays
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001          # Initial learning rate.
    LEARNING_MOMENTUM = 0.999     # The decay to use for the moving average.

    def __init__(self):
        """Set values of computed attributes."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[self.GPU_USAGE], True)
                tf.config.experimental.set_visible_devices(gpus[self.GPU_USAGE], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
