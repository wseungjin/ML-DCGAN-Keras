import tensorflow as tf
import numpy as np
import random, os
from glob import glob
from tensorflow.contrib import slim
import cv2


class ImageData:
    
    def __init__(self, img_size, channels):
        #64 X 64
        self.img_size = img_size
        #rgb 3
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def load_data(dataset_name) :
    x = glob(os.path.join("./dataset", dataset_name, '*.*'))
    return x