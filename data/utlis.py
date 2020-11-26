import os
import PIL
import numpy as np


class img_reader:

    def __init__(self, path):
        self.path = path

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
            img = np.array(PIL.Image.open(f"{os.path.join(self.path, self.img_list[i])}"))
            self.images[i] = img

    def thumbnail(self, *size):
        for i in self.images:
            i.thumbnail(*size)  #  preserve aspect ratio

    def resize(self, *size):
        for i in range(self.N):
            self.images[i] = self.images[i].resize(*size)

    @property
    def N(self):
        return len(self.img_list)
    
