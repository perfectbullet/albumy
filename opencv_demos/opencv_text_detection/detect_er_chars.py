#!/usr/bin/python

import sys
import os

import cv2 as cv
from opencv_text_detection.deeptextdetection import resize

'''
A simple demo script using the Extremal Region Filter algorithm described in:
Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n'
'''

img = cv.imread('109573.jpg')
img = resize(img, width=400)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 创建ERFilter对象时，允许隐式加载默认分类器。 returns a pointer to ERFilter::Callback
erc1 = cv.text.loadClassifierNM1('trained_classifierNM1.xml')
er1 = cv.text.createERFilterNM1(erc1)

erc2 = cv.text.loadClassifierNM2('trained_classifierNM2.xml')
er2 = cv.text.createERFilterNM2(erc2)

regions = cv.text.detectRegions(gray, er1, er2)

# Visualization
rects = [cv.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
for rect in rects:
    cv.rectangle(img, rect[0:2], (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 0), 1)
# for rect in rects:
#     cv.rectangle(img, rect[0:2], (rect[0]+rect[2], rect[1]+rect[3]), (255, 255, 255), 1)

cv.namedWindow("Text detection result")
cv.imshow("Text detection result", img)
cv.waitKey(0)
