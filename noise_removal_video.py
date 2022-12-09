import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('Marvels Spider-Man – Be Greater Extended Trailer _ PS4.mp4')

# criar uma lista com os 5 primeiros frames do vídeo
img = [cap.read()[1] for i in range(5)]

# converter para escala de cinza
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]

# conveter para float64
gray = [np.float64(i) for i in gray]

# criando ruído com uma variancia de 25
noise = np.random.randn(*gray[1].shape)*10

# adicionando o ruído nas imagens
noisy = [i+noise for i in gray]

# Converter para int8
noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

# Removendo o ruído do terceiro frame
dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
plt.subplot(131),plt.imshow(gray[2],'gray')
plt.subplot(132),plt.imshow(noisy[2],'gray')
plt.subplot(133),plt.imshow(dst,'gray')
plt.show()