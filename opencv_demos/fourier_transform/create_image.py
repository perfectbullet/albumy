#!/usr/bin/python
# coding=utf-8

import numpy as np
import cv2


def main():
    black_img = np.zeros((128, 128))
    black_img[50][0] = 255
    print(cv2.imwrite('black1.jpg', black_img))


if __name__ == '__main__':
    main()
