import tensorflow as tf
import tensorflow.keras as kr

class discriminator:

    """
    encapsulated class for generator 
    """

    def __init__(self, 
                model,  # a generator model for training
                ):
        super().__init__()

        self.model = _discriminator_model()
        self.loss = kr.losses.BinaryCrossentropy(from_logits=True)
        

    
    def training(self):
        return 

    def test(self):
        return 



class _discriminator_model(kr.Model):

    def __init__(self):
        super().__init__()
        self.l = kr.Sequential()

        self.l.add(kr.layers.Conv2D(filters=64, kernel_size=(5,5), 
                        strides=(2,2), padding="same", data_format="channels_last"))  # [batch_shape, rows, cols, channels] if data_format='channels_last'.

        self.l.add(kr.layers.LeakyReLU())
        self.l.add(kr.layers.Dropout(0.3))

        self.l.add(kr.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.l.add(kr.layers.LeakyReLU())
        self.l.add(kr.layers.Dropout(0.3))

        self.l.add(kr.layers.Flatten())
        self.l.add(kr.layers.Dense(1))
    
    def call(self, x):
        """
        Args:
            x: tensor; shape -> [batch_size, height, width, channel_num]
        """
        return self.l(x);  # [batch_size, 1]



if __name__ == "__main__":

    """
    generate toy data to test the discriminator
    """

    # [batch_size, height, width, channel_num]
    data = tf.random.normal(shape=(10, 16, 16, 1), stddev=10)
    noise = tf.random.uniform(shape=(10, 16, 16, 1))


    m = _discriminator_model()
    print(m(data).shape)


    
