import numpy as np
import cv2
from matplotlib import pyplot as plt


def deconv(channel):
    f_blueblur = np.fft.fft2(channel)

    # 讀取 .npy 文件
    f_psf = np.load('f_psf.npy')

    ff_blueorg = f_blueblur / f_psf

    inverse_fourier_transform = np.real(np.fft.ifft2(ff_blueorg))


    org_deconvolved = cv2.normalize(inverse_fourier_transform, None, 0, 255, cv2.NORM_MINMAX)
    return org_deconvolved

blur = cv2.imread("input1.bmp")

blue_channel, green_channel, red_channel = cv2.split(blur)

blue = deconv(blue_channel)
green = deconv(green_channel)
red = deconv(red_channel)



org_image = cv2.merge([blue, green, red])
# 創建新的圖形
plt.figure()

plt.imshow(org_image/255, cmap='viridis'), plt.title('org_image')
plt.savefig('reverseblur.png')
plt.show()