import time
import os

import cv2
import numpy as np



if __name__ == '__main__':
    from os.path import join
    # 正样本的标签为1, 负样本的标签为0
    base_train_dir = '/disk_workspace/train_data_for_svm/tgse_train'
    # indirs = [join(base_train_dir, pa) for pa in ['zzjg_zm', 'zzjg_inner', 'zzjg_lt', 'zzjg_bm']]
    # indirs = [join(base_train_dir, pa) for pa in ['套管双耳上部_包含多个', '套管双耳下部_包含多个']]
    # inlabels = [1, -1]
    # inlabels = [1, 2, 3, -1]
    outpath = join(base_train_dir, 'tgse-20191114-two.dat')  # 模型输出目录

    # hog特征描述器
    # 参数图解: https://blog.csdn.net/qq_26898461/article/details/46786285
    # 参数说明: winSize:窗口大小, blockSize:块大小, blockStride:块滑动增量, cellSize:胞元大小, nbins:梯度方向数目
    descriptor = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    # 拟合
    # datas, labels = getDataset(indirs, inlabels, descriptor, size=(64, 64), winStride=(8, 8), padding=(0, 0))
    # print('datas.shape={}, labels.shape={}'.format(datas.shape, labels.shape))
    # svm = cv2.ml.SVM_create()
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setC(2.67)
    # svm.setGamma(5.383)
    # svm.train(datas, cv2.ml.ROW_SAMPLE, labels)
    # print('outpath={}'.format(outpath))
    # svm.save(outpath)

    # 开始测试, 测试数据和拟合的数据不能有重复, 有重复的测试结果不能说明问题
    svmer = cv2.ml.SVM_load(outpath)
    test_dirs = [join(base_train_dir, pa) for pa in ['套管双耳上部_包含多个_test', '套管双耳下部_包含多个_test']]
    test_lables = [1, -1]
    test_des_data, test_labels = getDataset(test_dirs, test_lables, descriptor, size=(64, 64), winStride=(8, 8), padding=(0, 0))
    test_query_data = np.array(test_des_data)  #
    ret, responses = svmer.predict(test_query_data)     # ret

    # Check Accuracy
    mask = test_labels == responses.reshape(responses.shape[0])
    correct = np.count_nonzero(mask)
    acc = correct / mask.size
    print('test_labels={}, responses.shape={}, mask.shape={}, acc={}'
          .format(test_labels.shape, responses.shape, mask.shape, acc))

#
#
# self.model.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels, 10,
#                      cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_C),
#                      cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_GAMMA),
#                      cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_P),
#                      cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_NU),
#                      cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_COEF),
#                      cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_DEGREE),
#                      False
#                      )