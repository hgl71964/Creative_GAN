import tensorflow as tf
import tensorflow.keras as kr


class generator:

    """
    encapsulated class for generator 
    """

    def __init__(self, 
                model,  # a generator model for training
                ):
        super().__init__()

        self.model = model

    
    def training(self):
        return 

    def test(self):
        return 



class _generator_model(kr.Model):

    def __init__(self):
        super().__init__()

    
    def call(self, x):
        return x;

    




if __name__ == "__main__":

    """
    generate toy data to test the generator
    """

    
