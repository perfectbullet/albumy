#!/usr/bin/python
# coding=utf-8

import time
import os

import cv2
import numpy as np


# hog特征描述器
# 参数图解: https://blog.csdn.net/qq_26898461/article/details/46786285
# 参数说明: winSize:窗口大小, blockSize:块大小, blockStride:块滑动增量, cellSize:胞元大小, nbins:梯度方向数目
descriptor = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)


# 获取数据集
# 参数: datadirs:数据目录,  labels:数据目录对应的标签, descriptor:特征描述器, size:图片归一化尺寸(通常是2的n次方, 比如(64,64)), kwargs:描述器计算特征的附加参数
# 返回值, descs:特征数据, labels:标签数据
def getDataset(datadirs, labels, descriptor, size, **kwargs):
    # 获取训练数据
    # 参数: path:图片目录,  label:图片标签,  descriptor:特征描述器, size:图片归一化尺寸(通常是2的n次方, 比如(64,64)), kwargs:描述器计算特征的附加参数
    # 返回值: 图像数据, 标签数据
    def getDatas(path, label):
        datas = []
        for root, dirs, files in os.walk(path):
            for fname in files:
                lowname = fname.lower()
                if not lowname.endswith('.jpg') and not lowname.endswith('.png') and not lowname.endswith('.bmp'): continue
                imgpath = os.path.join(root, fname)
                gray = cv2.imread(imgpath, 0)
                if gray is None or len(gray) < 10:
                    continue
                desc = descriptor.compute(cv2.resize(gray, size,interpolation=cv2.INTER_AREA), **kwargs).reshape((-1))
                datas.append(desc)
        return np.array(datas), np.full((len(datas)), label, dtype=np.int32)

    descs, dlabels = None, None
    for path, label in zip(datadirs, labels):
        if descs is None:
            descs, dlabels = getDatas(path, label)
        else:
            ds, ls = getDatas(path, label)
            descs, dlabels = np.vstack((descs, ds)), np.hstack((dlabels, ls))
    return descs, dlabels


class HOG_Detector(object):
    """
    计算图像的 hog descriptor
    """
    def __init__(self, lable2code, svm_data_path='svm_data_jg_20191112.dat', win_size=(64, 64), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.win_size = win_size  # 窗口大小, 就是图像的shape
        self.block_size = block_size  # block，
        self.block_stride = block_stride  # block step
        self.cell_size = cell_size  # cell size, 梯度直方图采样框
        self.nbins = nbins  # hist bins, 和梯度直方图的粒度有关
        # HOG 描述符计算器
        self.hog_det = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)
        self.svmer = cv2.ml.SVM_load(svm_data_path)      # 向量机
        self.lable2code = lable2code    # like {1: 'C_ZZ_JG_ZM', -1: 'C_ZZ_JG_BM'}

    def _get_hog_data(self, gray_img):
        """
        计算图像 hog descriptor, 先缩小， 在计算 hog descriptor
        :param gray_img:    构件灰度图
        :return: ndarray
        """
        t1 = time.time()
        dst = cv2.resize(gray_img, self.win_size, interpolation=cv2.INTER_AREA)
        t2 = time.time()
        print('_reshape_image T={}'.format(t2 - t1))
        return self.hog_det.compute(dst, winStride=self.block_stride, padding=(0, 0))

    def get_result(self, gary_img):
        """
        获取一个分类的结果
        :param gary_img: 构件灰度图
        :return: 一个分类的结果 （int->code, str->name）
        """
        query_data = self._get_hog_data(gary_img)
        query_data_pi = np.array([query_data, ])    #
        qurey_re = self.svmer.predict(query_data_pi)[1]
        return int(qurey_re), self.lable2code.get(int(qurey_re))
