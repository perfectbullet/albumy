#!/usr/bin/python
# coding=utf-8

import time
import sys
import pickle
import os
import shutil
import random
from os import walk
from os.path import join

import cv2

from svm_train.svm_mode_and_descriptor import HOG_Detector


def loadPicklePath(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def write_to(impath, img):
    """
    保存图片
    :param impath:
    :param img:
    :return: None
    """
    su = cv2.imwrite(impath, img)
    if not su:
        print('failed fname={}'.format(impath))
        sys.exit(1)


def pick_by_comscore(base_dir):
    # 根据分数把样本分到不同的夹子中
    # base_dir = '/disk_workspace/train_data_for_svm/tgde_samples20191126/C_TGDE_PT_ZM_BM20191202'
    rootDir = join(base_dir, 'pickledata')
    comp_img_sclt06 = join(base_dir, 'comp_img_sclt06')  # 分数小于0.6
    comp_img_scgt06 = join(base_dir, 'comp_img_scgt06')  # 分数大于0.6
    comp_img_scgt08 = join(base_dir, 'comp_img_scgt08')  # 分数大于0.8
    comp_img_scgt09 = join(base_dir, 'comp_img_scgt09')  # 分数大于0.9

    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        print('root={}, len files={}'.format(root, len(files)))
        for f in files:
            if '.txt' not in f:
                continue
            # print('fname={}'.format(f))
            task = loadPicklePath(os.path.join(rootDir, f))
            task.xml.removeDupAreas(0.2)
            careas = task.xml.getAreasByFlag("C_DWHXJ")  # 获取名称以C_开头的，cArea区域
            if not careas:
                continue
            task.load()
            for ca in careas:
                ca_img = task.img.getImgROI(ca)
                ca_score = ca.score
                fname = 'sc_{:.3f}_{}_{}'.format(ca_score, str(ca), '.jpg')
                if ca_score <= 0.6:
                    write_to(os.path.join(comp_img_sclt06, fname), ca_img)
                if ca_score > 0.6:
                    write_to(os.path.join(comp_img_scgt06, fname), ca_img)
                if ca_score > 0.8:
                    write_to(os.path.join(comp_img_scgt08, fname), ca_img)
                if ca_score > 0.9:
                    write_to(os.path.join(comp_img_scgt09, fname), ca_img)


def random_sample(img_dir, sample_rat=0.2):
    # 对根目录下的样本文件夹随机采样
    dirs = [join(img_dir, d) for d in os.listdir(img_dir) if os.path.isdir(join(img_dir, d))]
    for d in dirs:
        if d.endswith('test') and len(os.listdir(d)) > 10:
            print('已经 random sample 过了')
            return None
    for root, dirs, files in walk(img_dir):
        if len(files) < 10:
            continue
        if root.endswith('test'):
            print(root)
            continue
        # random pick
        sample_files = random.sample(files, int(sample_rat * len(files)))
        print('root={}, files_len={}, sample_files_len={}'.format(root, len(files), len(sample_files)))
        test_root = root + '_test'
        # move to test dirs
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        for fname in sample_files:
            shutil.move(join(root, fname), join(test_root, fname))


def pick_sample_by_model(samp_dir, lable2code, model_path):
    # 使用模型挑选样本
    hog_dtector = HOG_Detector(lable2code, model_path)
    for lb in lable2code.items():
        make_dir(join(samp_dir, lb[1]))
    for img_name in os.listdir(samp_dir):
        fpath = join(samp_dir, img_name)
        if fpath.endswith('.jpg'):
            query_img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            t1 = time.time()
            result = hog_dtector.get_result(query_img)
            t2 = time.time()
            print('result={}, T={}'.format(result, t2 - t1))
            shutil.move(fpath, join(samp_dir, result[1], img_name))


if __name__ == '__main__':
    samp_dir = '/disk_workspace/train_data_for_svm/zzjg_train_20191225/train_data_two'
    random_sample(samp_dir)
    
    # 模型挑选样本
    # samp_dir = '/disk_workspace/train_data_for_svm/tgde_samples20191126/C_TGDE_PT_PART_ONE_TWO_1217/C_TGDE_PT_ZM_TWO_PART'
    # base_train_dir = '/disk_workspace/train_data_for_svm/tgde_samples20191126/C_TGDE_PT_PART_ONE_TWO_1217/'
    # lable2code = {1: 'C_TGDE_PT_ZM_ONE_PART', -1: 'C_TGDE_PT_ZM_TWO_PART'}
    # model_path = join(base_train_dir, 'tgde-one-two-part.dat')  # 模型输出目录
    # pick_sample_by_model(samp_dir, lable2code, model_path)
    pass
