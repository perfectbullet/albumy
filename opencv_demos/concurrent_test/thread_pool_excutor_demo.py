#!/usr/bin/python
# coding=utf-8

import time
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import cv2

from get_blurry_res.get_blurry_res import get_blurry_res


def imread(impath):
    return cv2.imread(impath, cv2.IMREAD_GRAYSCALE)


def get_blurry(impath):
    print('impath in get_blurry={}'.format(impath))
    imdata = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    return get_blurry_res(imdata)


def get_blurry_mulpro(im_dir):
    """
    use ProcessPoolExecutor
    :param im_dir:
    :return:
    """
    # imread_gen = [imread(os.path.join(im_dir, im_name)) for im_name in os.listdir(im_dir)]
    impath_gen = [os.path.join(im_dir, im_name) for im_name in os.listdir(im_dir)]
    # for imdata in imread_gen:
    #     print('gray_img={}'.format(type(imdata)))
    with ProcessPoolExecutor(4) as executor:
        future_tks = [executor.submit(get_blurry, (impath)) for impath in impath_gen]
        start_time = time.time()
        for future in as_completed(future_tks):
            im_data = future.result()
            res = 'thread im_data={}'.format(im_data,)
            print(res)
        print("last time is: {}".format(time.time()-start_time))


def get_blurry_onepro(im_dir):
    """
    use one process
    :param im_dir:
    :return:
    """
    impath_gen = [os.path.join(im_dir, im_name) for im_name in os.listdir(im_dir)]
    # for imdata in imread_gen:
    #     print('gray_img={}'.format(type(imdata)))
    with ProcessPoolExecutor(4) as executor:
        future_tks = [executor.submit(get_blurry, (impath)) for impath in impath_gen]
        start_time = time.time()
        for future in as_completed(future_tks):
            im_data = future.result()
            res = 'thread im_data={}'.format(im_data, )
            print(res)
        print("last time is: {}".format(time.time() - start_time))


if __name__ == '__main__':
    # im_dir = '/disk_workspace/train_data_for_svm/tgde_samples20191126/C_TGDE_GT/clear'
    im_dir = '/disk_workspace/train_data_for_svm/test_test'
    
    t1 = time.time()
    get_blurry_mulpro(im_dir)
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     # Start the load operations and mark each future with its path
    #     # future_to_imread = {executor.submit(imread, impath): impath for impath in impaths}
    #     imread_futures = [executor.submit(imread, (os.path.join(im_dir, im_name))) for im_name in os.listdir(im_dir)]
    #     # imread_futures[0].type=<class 'concurrent.futures._base.Future'>
    #
    #     for future in as_completed(imread_futures):
    #         # print('future type={}'.format(type(future)))
    #         # future type=<class 'concurrent.futures._base.Future'>
    #         try:
    #             im_data = future.result()
    #             blurry_thr = get_blurry_res(im_data)
    #             res = 'thread data.shape={}, blurry_thr={}'.format(im_data.shape, blurry_thr)
    #             print(res)
    #         except Exception as exc:
    #             print('generated an exception: %s' % (exc, ))
    # t2 = time.time()
    # imread_gen = (imread(os.path.join(im_dir, im_name)) for im_name in os.listdir(im_dir))
    # for im_data in imread_gen:
    #     blurry_thr = get_blurry_res(im_data)
    #     res = 'generator data.shape={}, blurry_thr={}'.format(im_data.shape, blurry_thr)
    #     print(res)
    # t3 = time.time()
    # for im_name in os.listdir(im_dir):
    #     im_data = imread(os.path.join(im_dir, im_name))
    #     blurry_thr = get_blurry_res(im_data)
    #     res = 'normal data.shape={}, blurry_thr={}'.format(im_data.shape, blurry_thr)
    #     print(res)
    # t4 = time.time()
    # print('task len={} T1={}  T2={}  T3={}'.format(len(os.listdir(im_dir)), t2 - t1, t3 - t2, t4 - t3))
