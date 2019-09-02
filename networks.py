import os.path 
import numpy as np 
from ops import *
from utills import *
from keras.models import * 
from keras.layers import * 
from keras.optimizers import * 
import keras.backend as K 
import matplotlib.pyplot as plt
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch


class DCGAN:
    def __init__(self,args):
        self.model_name = "DCGAN"  # name for checkpoint
        self.phase = args.phase
        
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.model_dir = args.model_dir
        
        self.dataset_name = args.dataset

        self.epoch = args.epoch
        self.iteration = args.iteration
        
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr


        
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq        
        
        self.img_size = args.img_size
        
        self.ch = args.ch        
        self.gan_type = args.gan_type
        self.z_dim = args.z_dim
        
        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num
        
        self.c_dim = 3
        self.data = load_data(dataset_name=self.dataset_name)
        self.custom_dataset = True

        self.dataset_num = len(self.data)
        
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        
        
        
        
        print() 

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# img_size : ", self.img_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print("##### Generator #####")
        print("# learning rate : ", self.g_learning_rate)

        print()

        print("##### Discriminator #####")
        print("# learning rate : ", self.d_learning_rate)    
        
        
    def generator(self):
        g_model = Sequential()
        
        g_model.add(Conv2D(self.ch*2,kernel_size=(3,3),strides=1,padding='same',input_shape=self.input_shape))
        g_model.add(BatchNormalization())
        g_model.add(ReLU())
        
        g_model.add(Conv2D(self.ch*2,kernel_size=(3,3),strides=2,padding='same'))
        g_model.add(BatchNormalization())
        g_model.add(ReLU())
        
        g_model.add(Conv2D(self.ch//2,kernel_size=(3,3),strides=2,padding='same'))
        g_model.add(BatchNormalization())
        g_model.add(ReLU())
        
        g_model.add(Conv2D(self.ch//2,kernel_size=(3,3),strides=2,padding='same'))
        g_model.add(BatchNormalization())
        g_model.add(ReLU())
        
        return g_model

    def discriminator(self):
        d_model=Sequential()
        
        d_model.add(Conv2D(self.ch*2,kernel_size=(3,3),strides=1,padding='same',input_shape=self.input_shape))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        
        d_model.add(Conv2D(self.ch*2,kernel_size=(3,3),strides=2,padding='same'))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        
        d_model.add(Conv2D(self.ch//2,kernel_size=(3,3),strides=2,padding='same'))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        
        d_model.add(Conv2D(self.ch//2,kernel_size=(3,3),strides=2,padding='same'))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        
        d_model.add(Flatten())
        d_model.add(Dense(1))
        d_model.add(Activation('softmax'))

        
        return d_model

    def build_model(self):
        
        Image_Data_Class = ImageData(self.img_size, self.c_dim)
        inputs = tf.data.Dataset.from_tensor_slices(self.data)

        gpu_device = '/gpu:0'
        self.inputs = inputs.\
            apply(shuffle_and_repeat(self.dataset_num)).\
            apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))
            
        numpy_datas=self.inputs.numpy()    
            
        print(numpy_datas.shape)    
        self.input_shape=self.inputs.shape
                
        
        self.g_model=self.generator()
        self.d_model=self.discriminator()
        
        self.d_optimizer = Adam(lr=self.d_learning_rate)
        self.d_model.compile(self.d_optimizer,loss="binary_crossentropy")
        
        self.g_optimizer = Adam(lr=self.g_learning_rate)
        self.g_model.compile("SGD",loss="binary_crossentropy")

        self.d_model.trainable = False
        self.g_d_model = Sequential
        self.g_d_model.add(self.g_model)
        self.g_d_model.add(self.d_model)
        self.g_d_model.compile(self.g_optimizer,loss="binary_crossentropy")



    def train(self):
        x_train = self.inputs
        BATCH_SIZE=self.batch_size
        
    
        # Some parameters. 
    
        for epoch in range(self.epoch):
            print("Epoch is", epoch)
            print("Number of batches", int(x_train.shape[0]/BATCH_SIZE))
            for index in range(int(x_train.shape[0]/BATCH_SIZE)):
                noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
                image_batch = x_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                generated_images = self.g_model.predict(noise, verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                self.d_loss = self.d_model.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, self.d_loss))
                noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
                self.d_model.trainable = False
                self.g_loss = self.g_d_model.train_on_batch(noise, [1] * BATCH_SIZE)
                self.d_model.trainable = True
                print("batch %d g_loss : %f" % (index, self.g_loss))
                if epoch % 10 == 0:
                    self.g_model.save_weights('generator', True)
                    self.d_model.save_weights('discriminator', True)


        
        
    # def test(self):
        