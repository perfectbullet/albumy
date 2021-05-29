#!/usr/bin/python
# coding=utf-8

import time

import cv2


if __name__ == '__main__':
    def loop():
        n = 0
        while True:
            n += 1
            im = cv2.imread('6-9060_2082_742.jpg')
            yield im
            # yield n
    
    lp_gn = loop()
    t1 = time.time()
    for idx, img in enumerate(lp_gn):
        print(img.shape)
        if idx > 1000:
            lp_gn.close()   # 结束后不会引起异常
    t3 = time.time()
    print('T={}'.format(t3 - t1))
