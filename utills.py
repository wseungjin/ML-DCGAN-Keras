import tensorflow as tf
import numpy as np
import random, os
from glob import glob
from tensorflow.contrib import slim
import cv2
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
# from tqdm import tqdm 


class ImageData:
    
    def __init__(self, img_size, channels):
        #64 X 64
        self.img_size = img_size
        #rgb 3
        self.channels = channels

    # def image_processing(self, filename):
    #     x = tf.read_file(filename)
    #     x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
    #     img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
    #     img = tf.cast(img, tf.float32) / 127.5 - 1

    #     return img
    def load_data(self,dataset_name,flag=0):
        if flag==0:
            TRAIN_DIR = os.path.join("./dataset", dataset_name)
            print(TRAIN_DIR)
            training_data = []
            for img in (os.listdir(TRAIN_DIR)):
                # print(img)
                path = os.path.join(TRAIN_DIR,img)
                # print(path)
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                # print(img)
                img = cv2.resize(img, (self.img_size,self.img_size))
                training_data.append([np.array(img).astype('float32')])
            shuffle(training_data)
            x_train = np.vstack(training_data) / 255.0
            np.save(str(dataset_name) + '_train_data.npy', x_train)
            print(x_train.shape)
        else:
            x_train=np.load(str(dataset_name)+'_train_data.npy')

            
    
        return x_train


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# def load_data(dataset_name) :
#     x = glob(os.path.join("./dataset", dataset_name, '*.*'))
#     return x

