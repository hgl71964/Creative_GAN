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
    


"""  use matplotlib to process """
# simg = mpimg.imread(f"{os.path.join(data_path, img_list[0])}")
# print(simg.shape)  #  numpy data

# #  plot the data
# imgplot = plt.imshow(simg)
# plt.show()

"""use PIL package to resize"""
image = PIL.Image.open(f"{os.path.join(data_path, img_list[0])}")
# image.show()  #  plot image

print(image.format) 
print(image.mode) 
print(image.size) 
print(image.palette) 
new_image = image.resize((400, 400))
new_image.show()

image.thumbnail((400,400))  #  preserve aspect ratio
print(image.format) 
print(image.mode) 
print(image.size) 
print(image.palette) 
image.show()
