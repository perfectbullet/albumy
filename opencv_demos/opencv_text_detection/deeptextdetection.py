# -*- coding: utf-8 -*-
#!/usr/bin/python
import sys
import os
import cv2
import numpy as np


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    resize image
    """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def main():
    print('\nDeeptextdetection.py')
    print('       A demo script of text box alogorithm of the paper:')
    print('       * Minghui Liao et al.: TextBoxes: A Fast Text Detector with a Single Deep Neural Network https://arxiv.org/abs/1611.06779\n')

    # if (len(sys.argv) < 2):
    #     print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
    #     quit()

    if not os.path.isfile('TextBoxes_icdar13.caffemodel') or not os.path.isfile('textbox.prototxt'):
        print(" Model files not found in current directory. Aborting")
        print(" See the documentation of text::TextDetectorCNN class to get download links.")
        quit()

    img = cv2.imread('109573.jpg', 1)
    img = resize(img, width=400)
    print('img.shape={}'.format(img.shape))

    textSpotter = cv2.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")
    cp_img = img.copy()
    rects, outProbs = textSpotter.detect(cp_img)
    vis = img.copy()
    thres = 0.6

    for r in range(np.shape(rects)[0]):
        if outProbs[r] > thres:
            rect = rects[r]
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    cv2.namedWindow('Text detection result')
    cv2.imshow("Text detection result", vis)
    cv2.waitKey()


if __name__ == "__main__":
    main()
