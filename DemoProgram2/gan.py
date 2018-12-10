from keras.models import Sequential, Model
from keras.layers import *
from keras.datasets import mnist
from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np

# Generate testcase for y=x**2

class GAN():
    def __init__(self):
        #output shape
        self.data_shape = (2,1)

        optimizer = Adam(0.0002, 0.5)

        # Discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated output
        z = Input(shape=(100,))
        data = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated output(test case) as input and determines validity
        valid = self.discriminator(data)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates output => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)
        
        model = Sequential()

        model.add(Dense(128, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.data_shape), activation='tanh'))
        model.add(Reshape(self.data_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        data = model(noise)

        return Model(noise, data)

    def build_discriminator(self):

        data_shape = (2,1)
        
        model = Sequential()

        model.add(Flatten(input_shape=data_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        data = Input(shape=data_shape)
        validity = model(data)

        return Model(data, validity)
    
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = self.load_data()
        print(X_train.shape)
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 0.5) / 0.5
        X_train = np.expand_dims(X_train, axis=2)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # Prepare inputs for generator and discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            datasets = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new outputs
            gen_data = self.generator.predict(noise)


            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(datasets, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_data, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train discriminator more interations
            """
            for i in range(4):
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                datasets = X_train[idx]
                d_loss_real = self.discriminator.train_on_batch(datasets, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_data, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            """


            noise = np.random.normal(0, 1, (batch_size, 100))
            # label generated outputs
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated outputs
            if epoch % save_interval == 0:
                self.save_data(epoch)

    def save_data(self, epoch):
        number_to_gen = 100
        noise = np.random.normal(0, 1, (100, 100))
        gen_data = self.generator.predict(noise)
        outfile = open(str(epoch) + '.txt', 'w')
        for i in range(number_to_gen):
            a = gen_data[i][0][0] *0.5 + 0.5
            b = gen_data[i][1][0] *0.5 + 0.5
            outfile.write(str(a) + ',' + str(b) + '\n' )
        outfile.close
        
        

    def load_data(self):
        my_data = np.genfromtxt('training_data.txt', delimiter=',')
        return my_data



if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=20000, batch_size=16, save_interval=500)
