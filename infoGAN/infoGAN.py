from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


class InfoGAN(object):
    """docstring for InfoGAN"""
    def __init__(self):
        # define the size of image (28 * 28 * 1)
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # define the size of latent code, acutally, the latent should contain the ratation and thickness information, now it is the number of classes
        self.num_classes = 10
        self.latent_dim = 72

        optimizer = Adam(0.0002,0.5)

        losses = ['binary_crossentropy', self.mutual_info_loss]
        # define the discrinimator and Q network, Q is a classifier
        self.discriminator, self.auxilliary = self.build_disk_and_qnet()

        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer, metrics=['accuracy'])

        self.auxilliary.compile(loss=[self.mutual_info_loss], optimizer=optimizer, metrics=['accuracy'])

        # define the generator
        self.generator = self.build_generator()
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)
        # only train the generator
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        target_label = self.auxilliary(img)# output of the Qnet is label
        self.combined = Model(gen_input, [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)



    def build_generator(self):
        '''
        Layer (type)                 Output Shape              Param #
        =================================================================
        dense_2 (Dense)              (None, 6272)              457856
        _________________________________________________________________
        reshape_2 (Reshape)          (None, 7, 7, 128)         0
        _________________________________________________________________
        batch_normalization_4 (Batch (None, 7, 7, 128)         512
        _________________________________________________________________
        up_sampling2d_3 (UpSampling2 (None, 14, 14, 128)       0
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 14, 14, 128)       147584
        _________________________________________________________________
        activation_3 (Activation)    (None, 14, 14, 128)       0
        _________________________________________________________________
        batch_normalization_5 (Batch (None, 14, 14, 128)       512
        _________________________________________________________________
        up_sampling2d_4 (UpSampling2 (None, 28, 28, 128)       0
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 28, 28, 64)        73792
        _________________________________________________________________
        activation_4 (Activation)    (None, 28, 28, 64)        0
        _________________________________________________________________
        batch_normalization_6 (Batch (None, 28, 28, 64)        256
        _________________________________________________________________
        conv2d_5 (Conv2D)            (None, 28, 28, 1)         577
        _________________________________________________________________
        activation_5 (Activation)    (None, 28, 28, 1)         0
        =================================================================

        '''
        model = Sequential()
        model.add(Dense(128*7*7, activation='relu', input_dim=self.latent_dim))
        model.add(Reshape((7,7,128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64,kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels,kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        gen_input = Input(shape=(self.latent_dim,))
        img = model(gen_input)
        model.summary()
        return Model(gen_input, img)

    def build_disk_and_qnet(self):
        # it is inverse of the generator
        img = Input(shape=self.img_shape)
        model = Sequential()
        model.add(Conv2D(64,kernel_size=3, strides=2, input_shape=self.img_shape,padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512,kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.summary()
        img_embedding = model(img)
        validity = Dense(1, activation='sigmoid')(img_embedding)
        q_net = Dense(128, activation='relu')(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        return Model(img, validity), Model(img, label)




    def mutual_info_loss(self, c, c_given_x):
        # equation 4
        eps = 1e-8
        conditional_entropy = K.mean(-K.sum(K.log(c_given_x+eps)*c, axis = 1))
        entropy = K.mean(-K.sum(K.log(c+eps)*c, axis=1))
        return conditional_entropy + entropy

    def sample_generator_input(self, batch_size):
        sampled_noise = np.random.normal(0,1,(batch_size,62))
        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1,1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)# one-hot encode
        return sampled_noise, sampled_labels

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, y_train), (_,_) = mnist.load_data()
        X_train = (X_train.astype(np.float32)-127.5)/127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1,1)
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        for epoch in range(epochs):
            # select a random batch images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])
            print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r,c = 10, 10
        fig, axs = plt.subplots(r,c)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(r)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5
            for j in range(r):
                axs[j,i].imshow(gen_imgs[j,:,:,0], cmap='gray')
                axs[j,i].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()


    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    infogan = InfoGAN()
    infogan.train(epochs=5000,batch_size=128, sample_interval=50)
