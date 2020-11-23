import tensorflow as tf
import tensorflow.keras as kr

class generator:

    """
    encapsulated class for generator 
    """

    def __init__(self, 
                ):
        super().__init__()

        self.model = _generator_model()

    
    def training(self):
        return 

    def test(self):
        return 

class _generator_model(kr.Model):

    """
    Start with a Dense layer that takes a seed (random noise) as input, then upsample several times until you reach the desired image size of 28x28x1
    """

    def __init__(self):
        super().__init__()

        self.l = kr.Sequential()
        self.l.add(kr.layers.Dense(7*7*256, use_bias=False))
        self.l.add(kr.layers.BatchNormalization())
        self.l.add(kr.layers.LeakyReLU())

        self.l.add(kr.layers.Reshape((7, 7, 256)))  # self.l.output_shape == (None, 7, 7, 256)  Note: None is the batch size

        self.l.add(kr.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))  # self.l.output_shape == (None, 7, 7, 128)
        self.l.add(kr.layers.BatchNormalization())
        self.l.add(kr.layers.LeakyReLU())

        self.l.add(kr.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))  # self.l.output_shape == (None, 14, 14, 64)
        self.l.add(kr.layers.BatchNormalization())
        self.l.add(kr.layers.LeakyReLU())

        self.l.add(kr.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))  # self.l.output_shape == (None, 28, 28, 1)

    
    def call(self, x):
        return self.l(x);



if __name__ == "__main__":

    """
    generate toy data to test the generator
    """


    #  assume dataset has 10 images, each (28*28)
    noise = tf.random.uniform(shape=(10, 100))

    m = _generator_model()

    print(m(noise).shape)

    
