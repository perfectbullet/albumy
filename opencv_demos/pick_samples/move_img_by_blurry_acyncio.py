#!/usr/bin/python
# coding=utf-8

import os
import time
import asyncio
from shutil import copyfile, move
from os.path import join

from get_blurry_res.get_blurry_res import get_blurry_res
from asyncio_test.afile_read import aimread


async def move_img_by_blurry(im_name, blurry_dir, clear_dir, blurry_thr):
    """
    单个任务
    :param im_name:
    :param blurry_dir:
    :param clear_dir:
    :param blurry_thr:
    :return:
    """
    src_path = join(src_dir, im_name)
    print('src_path={}'.format(src_path))
    gray_im = await aimread(src_path)
    bl_res = get_blurry_res(gray_im)
    # bl_res = get_blurry_res(cv2.imread(src_path, cv2.IMREAD_GRAYSCALE))
    if bl_res > blurry_thr:
        dst_path = join(clear_dir, im_name)
    else:
        dst_path = join(blurry_dir, im_name)
    print('dst_path={}'.format(dst_path))
    move(src_path, dst_path)


def move_by_blurry(src_dir, blurry_dir, clear_dir, blurry_thr):
    """
    根据blurry 移动图片
    :return:
    """
    mv_tasks = [move_img_by_blurry(im_name, blurry_dir, clear_dir, blurry_thr)
                for im_name in os.listdir(src_dir) if im_name.endswith('.jpg')]
    print('len of mv_tasks={}'.format(len(mv_tasks)))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(mv_tasks))
    loop.close()
    
    
def refresh(src_dir, blurry_dir, clear_dir):
    for fn in os.listdir(blurry_dir):
        move(join(blurry_dir, fn), join(src_dir, fn))
    for fn in os.listdir(clear_dir):
        move(join(clear_dir, fn), join(src_dir, fn))

    
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


if __name__ == '__main__':
    # src_dir = '/disk_workspace/train_data_for_svm/dwhx_20191122_dir/test_test'
    src_dir = '/disk_workspace/train_data_for_svm/dwhx_20191122_dir/comp_img_scgt06'
    blurry_dir = make_dir(join(src_dir, 'blurry'))
    clear_dir = make_dir(join(src_dir, 'clear'))
    blurry_thr = 2
    t1 = time.time()
    move_by_blurry(src_dir, blurry_dir, clear_dir, blurry_thr)
    t2 = time.time()
    print('T={}'.format(t2 - t1))
    refresh(src_dir, blurry_dir, clear_dir)
