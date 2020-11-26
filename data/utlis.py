import os
import PIL


class img_reader:

    def __init__(self, path):
        self.path = path
    

    def get_jpeg(self):
        self.img_list = []
        for i in os.listdir(f"{self.path}"):
            if i[-4:]=="jpeg":
                self.img_list.append(i)
    
    def plot(self, 
            n,  #  plot the number of image
            ):

        if n < len(self.N):
            image = PIL.Image.open(f"{os.path.join(self.path, self.img_list[n])}")
        else:
            raise ValueError("Not enough data points")

    @property
    def N(self):
        return len(self.img_list)
    
