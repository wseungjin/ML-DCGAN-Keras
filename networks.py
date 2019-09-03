import os.path 
import numpy as np 
from ops import *
from utills import *
from keras.models import * 
from keras.layers import * 
from keras.optimizers import * 
import keras.backend as K 
import matplotlib.pyplot as plt
from PIL import Image

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
        self.custom_dataset = True

        # self.dataset_num = len(self.data)
        
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        
        
        
        
        print() 

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        # print("# dataset number : ", self.dataset_num)
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
        dropout = 0.5
        
        g_model.add(Dense(8 * 8 * 256, input_shape=(self.z_dim,)))
        
        g_model.add(BatchNormalization(momentum=0.8))
        g_model.add(ReLU())
        g_model.add(Reshape((8, 8, 256)))
        g_model.add(Dropout(dropout))

        g_model.add(Dropout(dropout))
        g_model.add(UpSampling2D())
        
        g_model.add(Conv2DTranspose(128, 5, padding='same'))
        g_model.add(BatchNormalization())
        g_model.add(ReLU())
        g_model.add(Dropout(dropout))
        
        g_model.add(UpSampling2D())
        
        g_model.add(Conv2DTranspose(64, 5, padding='same'))
        g_model.add(BatchNormalization(momentum=0.9))
        g_model.add(Activation('relu'))
        g_model.add(Dropout(dropout))

        g_model.add(UpSampling2D())
        
        g_model.add(Conv2DTranspose(32, 5, padding='same'))
        g_model.add(BatchNormalization())
        g_model.add(ReLU())
        g_model.add(Dropout(dropout))

        # g_model.add(UpSampling2D())
        
        g_model.add(Conv2DTranspose(self.c_dim, 5, padding='same')) 
        g_model.add(Activation('sigmoid'))

        
        return g_model

    def discriminator(self):
        d_model=Sequential()
        dropout = 0.5
        
        d_model.add(Conv2D(self.ch*2,kernel_size=(3,3),strides=1,padding='same',input_shape=self.input_shape))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        d_model.add(Dropout(dropout))
        
        d_model.add(Conv2D(self.ch*2,kernel_size=(3,3),strides=2,padding='same'))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        d_model.add(Dropout(dropout))
        
        d_model.add(Conv2D(self.ch//2,kernel_size=(3,3),strides=2,padding='same'))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        d_model.add(Dropout(dropout))
        
        d_model.add(Conv2D(self.ch//2,kernel_size=(3,3),strides=2,padding='same'))
        d_model.add(BatchNormalization())
        d_model.add(ReLU())
        d_model.add(Dropout(dropout))
        
        d_model.add(Flatten())
        d_model.add(Dense(1))
        d_model.add(Activation('softmax'))

        
        return d_model

    def build_model(self):
        
        Image_Data_Class = ImageData(self.img_size, self.c_dim)
        self.data = Image_Data_Class.load_data(dataset_name=self.dataset_name)

        
        self.input_shape=(64,64,3)
                        
        
        self.g_model=self.generator()
        self.g_model.summary()
        self.d_model=self.discriminator()
        self.d_model.summary()
        
        self.d_optimizer = Adam(lr=self.d_learning_rate)
        self.d_model.compile(self.d_optimizer,loss="binary_crossentropy")
        
        self.g_optimizer = Adam(lr=self.g_learning_rate)
        self.g_model.compile("SGD",loss="binary_crossentropy")

        self.d_model.trainable = False
        self.g_d_model = Sequential()
        self.g_d_model.add(self.g_model)
        self.g_d_model.add(self.d_model)
        self.g_d_model.summary()
        
        self.g_d_model.compile(self.g_optimizer,loss="binary_crossentropy")



    def train(self):
        x_train = self.data
        print(x_train.shape)
        BATCH_SIZE=self.batch_size
        
    
        # Some parameters. 
    
        for epoch in range(self.epoch):
            print("Epoch is", epoch)
            print("Number of batches", int(x_train.shape[0]/BATCH_SIZE))
            for index in range(int(x_train.shape[0]/BATCH_SIZE)):
                noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 128))
                image_batch = x_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                generated_images = self.g_model.predict(noise, verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                self.d_loss = self.d_model.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, self.d_loss))
                noise = np.random.uniform(-1, 1, (BATCH_SIZE, 128))
                self.d_model.trainable = False
                self.g_loss = self.g_d_model.train_on_batch(noise, [1] * BATCH_SIZE)
                self.d_model.trainable = True
                print("batch %d g_loss : %f" % (index, self.g_loss))
                if epoch % 10 == 0:
                    self.g_model.save_weights('generator', True)
                    self.d_model.save_weights('discriminator', True)
                    
        self.phase='test'


        
        
    def test(self):
        g = self.generator()
        g.compile(loss='binary_crossentropy', optimizer="SGD")
        g.load_weights('generator')
        for i in range(20):
            noise = np.random.uniform(-1, 1, (self.batch_size, 128))
            generated_images = g.predict(noise, verbose=1)
            image = image*255.0
            Image.fromarray(image.astype(np.uint8)).save("generated_image_"+str(i) +".png")
            
