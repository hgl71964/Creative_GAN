import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import PIL
import numpy as np


data_path = os.path.join(os.getcwd(), "data")

print("")
print()


img_list = []
for i in os.listdir(f"{data_path}"):
    if i[-4:]=="jpeg":
        img_list.append(i)
print(img_list)


"""  use matplotlib to process """
# simg = mpimg.imread(f"{os.path.join(data_path, img_list[0])}")
# print(simg.shape)  #  numpy data

# #  plot the data
# imgplot = plt.imshow(simg)
# plt.show()

"""use PIL package to resize"""
image = PIL.Image.open(f"{os.path.join(data_path, img_list[0])}")
# image.show()  #  plot image

# print(image.format) 
# print(image.mode) 
# print(image.size) 
# print(image.palette) 
# new_image = image.resize((400, 400))
# new_image.show()

# image.thumbnail((400,400))  #  preserve aspect ratio
# print(image.format) 
# print(image.mode) 
# print(image.size) 
# print(image.palette) 
# image.show()

print(image.size)

i = np.array(image)
print(i.shape)