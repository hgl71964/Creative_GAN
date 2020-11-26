import tensorflow as tf
import tensorflow.keras as kr
import matplotlib.pyplot as plt

class generator_base(kr.Model):

    """
    Start with a Dense layer that takes a seed (random noise) as input, then upsample several times until you reach the desired image size of 28x28x1
    """

    def __init__(self):
        super().__init__()

        self.l = kr.Sequential()
        self.l.add(kr.layers.Dense(7*7*256, use_bias=False))  #  batch size: subject to input noise batch size
        self.l.add(kr.layers.BatchNormalization())
        self.l.add(kr.layers.LeakyReLU())

        self.l.add(kr.layers.Reshape((7, 7, 256)))  # output_shape == (None, 7, 7, 256)  Note: None is the batch size

        self.l.add(kr.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))  # output_shape == (None, 7, 7, 128)
        self.l.add(kr.layers.BatchNormalization())
        self.l.add(kr.layers.LeakyReLU())

        self.l.add(kr.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))  # output_shape == (None, 14, 14, 64)
        self.l.add(kr.layers.BatchNormalization())
        self.l.add(kr.layers.LeakyReLU())

        
    
    def call(self, x, training = False):
        return self.l(x);


class generator_model(generator_base):
    def __init__(self, 
                out_channel_num = 1,
                ):
        super().__init__()
        self.l.add(kr.layers.Conv2DTranspose(out_channel_num, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))  # output_shape == (None, 28, 28, 1)
    
    def call(self, x, training = False):
        return self.l(x);


if __name__ == "__main__":

    """
    generate toy data to test the generator
    """

    #  assume dataset has 10 images, each (28*28)
    noise = tf.random.uniform(shape=(1, 100))

    g = generator_model()

    image = g(noise, training = False)

    print(image.shape)

    plt.imshow(image[0, :, :, 0])  # cmap='gray') -> for gray scale
    plt.show()


    
