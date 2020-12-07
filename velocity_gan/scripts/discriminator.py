import tensorflow as tf
import tensorflow.keras as kr

class discriminator_model(kr.Model):

    def __init__(self):
        super().__init__()

        self.l1 = (kr.layers.Conv2D(filters=64, kernel_size=(5,5), 
                        strides=(2,2), padding="same", data_format="channels_last"))  # [batch_size, rows, cols, channels] if data_format='channels_last'.

        self.a1 = (kr.layers.LeakyReLU())
        self.p1 = (kr.layers.Dropout(0.3))

        self.l2 = (kr.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.a2 = (kr.layers.LeakyReLU())
        self.p2 = (kr.layers.Dropout(0.3))

        self.d1 = (kr.layers.Flatten())
        self.d2 = (kr.layers.Dense(1))
    
    def call(self, x, training = False):
        """
        Args:
            x: tensor; shape -> [batch_size, height, width, channel_num]: e.g. (None, 28, 28, 1); notice this discriminator 
        """

        x = self.a1(self.l1(x))
        if training:
            x = self.p1(x)

        x = self.a2(self.l2(x))
        if training:
            x = self.p2(x)
        
        return self.d2(self.d1(x))  # [batch_size, 1]

if __name__ == "__main__":

    """
    generate toy data to test the discriminator
    """

    # [batch_size, height, width, channel_num]
    data = tf.random.normal(shape=(1, 16, 16, 1), stddev=10)
    noise = tf.random.uniform(shape=(1, 16, 16, 1))


    m = discriminator_model()
    print(m(data, training = False))


    
