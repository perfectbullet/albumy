#!/usr/bin/python
# coding=utf-8


import cv2

from svm_train.svm_mode_and_descriptor import getDataset


def ZZ_JG_FOUR():
    """
        使用 SVM 对构件进行四分类：
        """
    from os.path import join
    import numpy as np
    
    # 正样本的标签为1, 负样本的标签为0
    base_train_dir = '/disk_workspace/train_data_for_svm/zzjg_train_20191225/train_data'
    indirs = [join(base_train_dir, pa) for pa in ['C_ZZ_JG_BM', 'C_ZZ_JG_ZM', 'C_ZZ_JG_INNER', 'C_ZZ_JG_LT']]
    inlabels = [-1, 1, 2, 3]
    outpath = join(base_train_dir, 'jg-20191225-four.dat')  # 模型输出目录
    
    descriptor = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8),
                                   _nbins=9)
    
    # # 拟合
    datas, labels = getDataset(indirs, inlabels, descriptor, size=(64, 64), winStride=(8, 8), padding=(0, 0))
    print('datas.shape={}, labels.shape={}'.format(datas.shape, labels.shape))
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(datas, cv2.ml.ROW_SAMPLE, labels)
    print('outpath={}'.format(outpath))
    svm.save(outpath)
    
    # 开始测试, 测试数据和拟合的数据不能有重复, 有重复的测试结果不能说明问题
    svmer = cv2.ml.SVM_load(outpath)
    test_dirs = [join(base_train_dir, pa) for pa in
                 ['C_ZZ_JG_BM_test', 'C_ZZ_JG_ZM_test', 'C_ZZ_JG_INNER_test', 'C_ZZ_JG_LT_test']]
    test_lables = [-1, 1, 2, 3]
    test_des_data, test_labels = getDataset(test_dirs, test_lables, descriptor, size=(64, 64), winStride=(8, 8),
                                            padding=(0, 0))
    test_query_data = np.array(test_des_data)  #
    ret, responses = svmer.predict(test_query_data)  # ret
    responses = responses.ravel()
    
    # Check Accuracy
    mask = test_labels * responses
    correct = np.count_nonzero(mask > 0)
    acc = correct / mask.size
    print('test_labels={}, responses.shape={}, mask.shape={}, acc={}'
          .format(test_labels.shape, responses.shape, mask.shape, acc))


def ZZ_JG_TWO():
    """
        使用 SVM 对构件进行四分类：
    """
    from os.path import join
    import numpy as np
    
    # 正样本的标签为1, 负样本的标签为0
    base_train_dir = '/disk_workspace/train_data_for_svm/zzjg_train_20191225/train_data_two'
    indirs = [join(base_train_dir, pa) for pa in ['C_ZZ_JG_BM', 'C_ZZ_JG_ZM']]
    inlabels = [-1, 1]
    outpath = join(base_train_dir, 'jg-20191225-two.dat')  # 模型输出目录
    
    descriptor = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8),
                                   _nbins=9)
    
    # # 拟合
    datas, labels = getDataset(indirs, inlabels, descriptor, size=(64, 64), winStride=(8, 8), padding=(0, 0))
    print('datas.shape={}, labels.shape={}'.format(datas.shape, labels.shape))
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(datas, cv2.ml.ROW_SAMPLE, labels)
    print('outpath={}'.format(outpath))
    svm.save(outpath)
    
    # 开始测试, 测试数据和拟合的数据不能有重复, 有重复的测试结果不能说明问题
    svmer = cv2.ml.SVM_load(outpath)
    test_dirs = [join(base_train_dir, pa) for pa in ['C_ZZ_JG_BM_test', 'C_ZZ_JG_ZM_test']]
    test_lables = [-1, 1, 2, 3]
    test_des_data, test_labels = getDataset(test_dirs, test_lables, descriptor, size=(64, 64), winStride=(8, 8),
                                            padding=(0, 0))
    test_query_data = np.array(test_des_data)  #
    ret, responses = svmer.predict(test_query_data)  # retZZ_JG_FOUR
    responses = responses.ravel()
    
    # Check Accuracy
    mask = test_labels == responses
    correct = np.count_nonzero(mask)
    acc = correct / mask.size
    print('test_labels={}, responses.shape={}, mask.shape={}, acc={}'
          .format(test_labels.shape, responses.shape, mask.shape, acc))


if __name__ == '__main__':
    ZZ_JG_TWO()
    ZZ_JG_FOUR()


