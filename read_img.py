import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


data_path = os.path.join(os.getcwd(), "data")

print("")
print()


img_list = []
for i in os.listdir(f"{data_path}"):
    if i[-4:]=="jpeg":
        img_list.append(i)
print(img_list)

simg = mpimg.imread(f"{os.path.join(data_path, img_list[0])}")
print(simg.shape)  #  numpy data


#  plot the data
imgplot = plt.imshow(simg)
plt.show()
