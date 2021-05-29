#!/usr/bin/env python

'''
K-means clusterization sample.
Usage:
   kmeans.py

Keyboard shortcuts:
   ESC   - exit
   space - generate new distribution
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from ml.gaussian_mix import make_gaussians

def main():
    cluster_n = 5
    img_size = 512

    # generating bright palette, 分类的颜色
    colors = np.zeros((1, cluster_n, 3), np.uint8)
    colors[0, :] = 255
    colors[0, :, 0] = np.arange(0, 180, 180.0/cluster_n)
    # print(colors[0, :, 0])
    colors = cv.cvtColor(colors, cv.COLOR_HSV2BGR)[0]

    while True:
        print('sampling distributions...')
        points, ref_distrs = make_gaussians(cluster_n, img_size)     # points.shape=(2460, 2)   len(ref_distrs)=5

        term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
        ret, labels, centers = cv.kmeans(points, cluster_n, None, term_crit, 10, 0)
        #
        # retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
        # 1. data: It should be of np.float32 data type, and each feature should be put in a single column.
        # 2. nclusters(K) : Number of clusters required at end

        img = np.zeros((img_size, img_size, 3), np.uint8)
        for (x, y), label in zip(np.int32(points), labels.ravel()):
            c = list(map(int, colors[label]))

            cv.circle(img, (x, y), 1, c, -1)

        cv.imshow('kmeans', img)
        ch = cv.waitKey(0)
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
