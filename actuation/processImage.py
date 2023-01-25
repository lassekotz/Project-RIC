import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def process_image(img):
    img = cv.imread(img, 0)
    canny_image = cv.Canny(img, 100, 200)

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.subplot(122),plt.imshow(canny_image, cmap='gray')

    plt.show()


process_image('picture1.jpg')