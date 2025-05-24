import numpy as np
import tensorflow as tf


def set_gpu_growing():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return len(gpus)


def set_random_seed(seed=0):
    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    