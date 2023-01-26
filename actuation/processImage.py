import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from numpy import asarray
from PIL import Image

def process_image(img, displayEdges = True):
    imgCV = cv.imread(img, 0)
    img = Image.open(img)
    numpydata = asarray(img) # 3D rgb ndarray

    dims = numpydata.shape
    canny_image = cv.Canny(imgCV, 50, 10) #2D greyscale ndarray
    if displayEdges == True:
        new_img = np.copy(numpydata)
        for i in range(dims[0]):
            for j in range(dims[1]):
                if canny_image[i][j] == 255:
                    new_img[i][j][1] = 255

        plt.imshow(new_img, interpolation='none', aspect='auto')
        plt.show()

    return canny_image #numpy.ndarray type

def save_csv(image_array):
    array_shape = image_array.shape
    flattened_image = image_array.reshape(1, array_shape[0]*array_shape[1])
    print(flattened_image.shape)
    np.savetxt("../data/images.csv", flattened_image, delimiter=",")


##EXAMPLE:

image_array = process_image('picture1.jpg', False)
#save_csv(image_array)
plt.imshow(image_array)
plt.show()