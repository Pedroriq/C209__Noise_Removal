import numpy as np
import cv2
from matplotlib import pyplot as plt

# lendo a imagem
img = cv2.imread('image_1.jpg')

# função da biblioteca opencv para remover os ruidos
denoising = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(denoising)
plt.show()

