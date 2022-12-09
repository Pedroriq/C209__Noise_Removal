from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

mean = 0
var = 10
sigma = var ** 1

# aplicando o ruído no frame, incrimentando o sigma em cada repetição (5x)
for i in range(0,5):
    img2 = cv2.imread("Marvels_Spider-Man-Frame_0.png")
    gaussian = np.random.normal(mean, sigma + i, (720,1280,3)) #  np.zeros((224, 224), np.float32)
    noisy_image = img2 + gaussian
    cv2.imwrite("frames_noise/frame{}_noise.png".format(i), noisy_image)

# abrindo as 5 imagens, o mesmo frame com 5 intensidades de ruídos diferentes
frame_0 = np.array(Image.open('frames_noise/frame0_noise.png'))
frame_1 = np.array(Image.open('frames_noise/frame1_noise.png'))
frame_2 = np.array(Image.open('frames_noise/frame2_noise.png'))
frame_3 = np.array(Image.open('frames_noise/frame3_noise.png'))
frame_4 = np.array(Image.open('frames_noise/frame4_noise.png'))
# aplicando a média aritimética nas 5 imagens
frame_avg = ( frame_0.astype(np.float64)
            + frame_1.astype(np.float64)
            + frame_2.astype(np.float64)
            + frame_3.astype(np.float64)
            + frame_4.astype(np.float64)) / 5
frame_avg = frame_avg.astype(np.uint8)

# salvando a imagem
Image.fromarray(frame_avg).save('result/noise_removal.png')
