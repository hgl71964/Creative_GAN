import tensorflow as tf
import tensorflow.keras as kr
import matplotlib.pyplot as plt

class Encoder(kr.Model):

    def __init__(self):
        super().__init__()


    
    def call(self, x , training = False):

        return 


class Decoder(kr.Model):

    def __init__(self):
        super().__init__()


    
    def call(self, x , training = False):

        return 


class generator_model(kr.Model):

    def __init__(self,
                e: "Encoder",
                d: "Decoder",
                ):
        super().__init__()

        self.encoder  = e
        self.decoder = d

    def call(self, x, training = False):
        """
        Args:
            x: [bathc size, num_reciver, time samples, num_source]; e.g. in the paper (None, 32, 1000, 3);
        """
        return self.decoder(self.encoder(x))

    def inference(self, x): 
        return tf.cast(tf.linalg.normalize(self.l(x), axis=3)[0]*225, tf.uint8)  # output uint \in [0, 225]



        



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


    
