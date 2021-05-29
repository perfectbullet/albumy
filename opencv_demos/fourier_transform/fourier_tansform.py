#!/usr/bin/python
# coding=utf-8

import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def show_dft():
    img = cv.imread('black1.jpg', 0)
    t1 = time.time()
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]))
    print('dft.shape={}, dft_shift.shape={}, magnitude_spectrum.shape={}'
          .format(dft.shape, dft_shift.shape, magnitude_spectrum.shape))
    t2 = time.time()
    print('T={:.4f}'.format(t2 - t1))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    show_dft()
    #
    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))

    # rows, cols = img.shape
    # crow, ccol = int(rows/2), int(cols/2)
    #
    # # create a mask first, center square is 1, remaining all zeros
    # mask = np.zeros((rows,cols,2),np.uint8)
    # mask[crow-10:crow+10, ccol-10:ccol+10] = 1
    #
    # # apply mask and inverse DFT
    # fshift = dft_shift*mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = cv.idft(f_ishift)
    # img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
    # t2 = time.time()
    # print('T={:.4f}'.format(t2 - t1))
    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()
