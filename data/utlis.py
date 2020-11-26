import os
import PIL
import numpy as np
import tensorflow as tf


class img_reader:

    def __init__(self, path):
        self.path = path
        self.images = []

    def get_jpeg(self):
        self.img_list = []
        for i in os.listdir(f"{self.path}"):
            if i[-4:]=="jpeg":
                self.img_list.append(i)
    
    def plot(self, n:  "plot the number of image"):

        if n < self.N:
            image = PIL.Image.open(f"{os.path.join(self.path, self.img_list[n])}")
            image.show()
        else:
            raise ValueError("Not enough data points")
    
    def load(self):
        self.images = [None]*self.N
        for i in range(self.N):
            self.images[i] = PIL.Image.open(f"{os.path.join(self.path, self.img_list[i])}")
            

    def thumbnail(self, *size):
        for i in self.images:
            i.thumbnail(*size)  #  preserve aspect ratio

    def resize(self, *size):
        for i in range(self.N):
            self.images[i] = self.images[i].resize(*size)

    def to_tensor(self):
        if not self.images: 
            print("Haven't loaded images")
        else:
            for i in range(self.N):
                self.images[i] = tf.convert_to_tensor(np.array(self.images[i]))
    
    def batcher(self, batch_size):
        if isinstance(self.images[0], tf.Tensor):
            
            for batch in range(0, self.N, batch_size):
                yield (self.images[batch:min(batch + batch_size, self.N)])
        else:
            raise TypeError("haven't converted to tensor!")

    @property
    def N(self):
        return len(self.img_list)
    
