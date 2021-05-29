#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: root
# datetime:2019/11/28 下午2:40
# software: PyCharm
# filename: CvClassifier.py

# decoding:utf-8
import numpy as np
import cv2, os, sys
from data.utils import loadPicklePath, savePickle, getTestFiles, getModelPath
from PIL import Image
# import locale
# locale.setlocale(locale.LC_ALL, 'C')
import time
from sklearn.model_selection import train_test_split
from test.cv2imgshow import *
from data.utils import getModelPath
# import tesserocr as ocr
from data.logger import catch_execpt

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',',
                   converters={0: lambda ch: ord(ch) - ord('A')})  # 导入的字母特征数据，并将字母转化为数字类别
    samples, labels = a[:, 1:], a[:, 0]  # 将类别给labels，特征给samples
    return samples, labels


class LetterStatModel(object):
    class_n = 10
    train_ratio = 0.5

    def __init__(self, model_name=""):
        self.model_name = model_name
        self.model = None
        pass

    def getSavePath(self, flag):
        import hashlib
        md5_val = hashlib.md5(self.model_name.encode('utf8')).hexdigest()
        fname = md5_val[:5] + "-" +flag + ".dat"
        return getModelPath(fname)

    def load(self, flag):
        # with open(self.getSavePath(flag), "r") as fn:
        # #     self.model.read(fn)
        # fnode = cv2.FileStorage(self.getSavePath(flag), cv2.FILE_STORAGE_READ)
        # self.model.read(fnode.root())
        self.model.load(self.getSavePath(flag))
        print "fnode"

    def save(self, flag):
        self.model.save(self.getSavePath(flag))


    def train(self, images, labels):
        unique_labels = np.unique(np.array(labels))
        x_data = []
        y_data = []
        for x, y in zip(images, labels):
            hogarr = self.HogCompute(x)
            x_data.append(hogarr)
            y_data.append(list(unique_labels).index(y))
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        self.trainHog(x_data, y_data)

    def trainHog(self, hogs, labels):
        pass

    def predict_one(self, image):
        return self.predict([image, ])[0]

    def predict(self, images):
        x_data = []
        for x in images:
            hogarr = self.HogCompute(x)
            x_data.append(hogarr)
        x_data = np.array(x_data)
        return self.predictHog(x_data)

    def predictHog(self, hogs):
        label, res = self.model.predict(hogs, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        result = res.flatten().astype(np.float32)
        return result

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape  # 获取特征维数和特征个数
        new_samples = np.zeros((sample_n * self.class_n, var_n + 1), np.float32)
        new_samples[:, :-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:, -1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_labels(self, labels):
        sample_n = len(labels)
        new_labels = np.zeros(sample_n * self.class_n, np.int32)
        resp_idx = np.int32(labels + np.arange(sample_n) * self.class_n)
        new_labels[resp_idx] = 1
        return new_labels

    def HogCompute(self, img, winSize=(64, 64), blockSize=(64, 64), blockStride=(8, 8), cellSize=(32, 32),
                   nbins=9, winStride=(8, 8), padding=(8, 8)):
        """
        计算hog特征向量
        :param img:
        :param winStride:
        :param padding:
        :return:
        """
        if not hasattr(self, "_hogger"):
            self.__hogger = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        return self.__hogger.compute(img, winStride, padding).reshape((-1,))


class OCRModel(LetterStatModel):
    def __init__(self, model_name):
        super(OCRModel, self).__init__(model_name)
        self.labelsMap = {}  # 所有的类别

    def init_model(self):
        import locale
        locale.setlocale(locale.LC_ALL, 'C')
        from data.utils import getResPath
        # import pytesseract as ocr
        import tesserocr as ocr
        gDigitOCR_LINE = ocr.PyTessBaseAPI(path=getResPath(''), lang='eng')
        print('version of tesseract: {}'.format(ocr.tesseract_version()))
        gDigitOCR_LINE.Init(path=getResPath(''), lang='eng',
                            oem=ocr.OEM.TESSERACT_LSTM_COMBINED)  # 这里的配置除了path，其他别修改
        gDigitOCR_LINE.SetVariable("tessedit_char_whitelist", "".join([str(val) for val in self.labelsMap.keys()]))
        gDigitOCR_LINE.SetPageSegMode(ocr.PSM.SINGLE_LINE)  # 单行文本
        self.model = gDigitOCR_LINE

    def save(self, fn):
        fname = self.getSavePath(fn)
        import json
        with open(fname, 'w') as f:
            json.dump(self.labelsMap, f)

    def load(self, fn):
        import json
        fname = self.getSavePath(fn)
        with open(fname, 'r') as f:
            self.labelsMap = json.load(f)
        self.init_model()

    def train(self, imgs, labels):
        unique_labels = np.unique(np.array(labels))
        [self.labelsMap.setdefault(label, i) for i, label in enumerate(unique_labels)]

    @catch_execpt(returnParams=(None, -1.0))
    def predict_one(self, img, muti_num=False):
        pil_img = Image.fromarray(img)  # 转换 <PIL.Image.Image image mode=L>
        self.model.SetImage(pil_img)
        if muti_num:
            return self.model.MapWordConfidences()[0]
        else:
            # txs = self.model.GetUTF8Text()
            # txx = self.model.MapWordConfidences()
            # confidences = self.model.GetComponentImages(ocr.RIL.SYMBOL,  True)
            # 正则匹配所有的数字
            import tesserocr as ocr
            self.model.AllWordConfidences()
            ri = self.model.GetIterator()
            level = ocr.RIL.SYMBOL
            num_confs = []
            for r in ocr.iterate_level(ri, level):
                try:
                    symbol = r.GetUTF8Text(level)  # r == ri
                    conf = r.Confidence(level) / 100
                except:
                    continue
                if symbol not in self.labelsMap.keys():
                    continue
                num_confs.append((self.labelsMap.get(symbol), conf))
                # if symbol:
                #     print(u'symbol {}, conf: {}'.format(symbol, conf))
                # indent = False
                # ci = r.GetChoiceIterator()
                # for c in ci:
                #     if indent:
                #         print('\t\t ')
                #     print('\t- ')
                #     choice = c.GetUTF8Text()  # c == ci
                #     print(u'{} conf: {}'.format(choice, c.Confidence()))
                #     indent = True
                # print('---------------------------------------------')
            num_confs.sort(key=lambda item: item[1])
            if not num_confs: return np.nan
            return num_confs[-1][0]

    def predict(self, images):
        results = []
        for img in images:
            results.append(self.predict_one(img))
        return np.array(results, dtype=np.float32)


class RTrees(LetterStatModel):
    # 已经可以正常使用
    def __init__(self, model_name):
        super(RTrees, self).__init__(model_name)
        self.model = cv2.ml_RTrees().create()


    def load(self, flag):
        fname = self.getSavePath(flag)
        self.model = cv2.ml.RTrees_load(fname)
        print "RTree loaded"

    def trainHog(self, samples, labels):

        sample_n, var_n = samples.shape
        var_types = np.array([cv2.ml.VAR_NUMERICAL] * var_n + [cv2.ml.VAR_CATEGORICAL], np.uint8)
        # CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=10)
        train_model = cv2.ml_RTrees(varType=var_types, params=params).create()
        # self.model.train(samples, cv2.ml.ROW_SAMPLE, labels, varType=var_types, params=params)
        train_model.train(samples, cv2.ml.ROW_SAMPLE, labels)
        self.model = train_model
        print "RTrees train finished"



class KNearest(LetterStatModel):
    def __init__(self, model_name):
        super(KNearest, self).__init__(model_name)
        self.model = cv2.ml_KNearest().create()

    def trainHog(self, samples, labels):
        self.model.train(samples, labels)
        print "KNearest train finished"

    def load(self, flag):
        fname = self.getSavePath(flag)


    def predictHog(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, k=10)
        return results.ravel()


class Boost(LetterStatModel):
    def __init__(self, model_name):
        super(Boost, self).__init__(model_name)
        self.model = cv2.ml_Boost().create()

    def trainHog(self, samples, labels):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_labels = self.unroll_labels(labels)
        var_types = np.array([cv2.ml.VAR_NUMERICAL] * var_n + [cv2.ml.VAR_CATEGORICAL, cv2.ml.VAR_CATEGORICAL],
                             np.uint8)
        # CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
        params = dict(max_depth=5)  # , use_surrogates=False)
        self.model.train(new_samples, cv2.ml.ROW_SAMPLE, new_labels, varType=var_types, params=params)
        print "boost train finished"
        self.save(self.model_name+".data")
        print "boost save finished"

    def predictHog(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array([self.model.predict(s, returnSum=True) for s in new_samples])
        pred = pred.reshape(-1, self.class_n).argmax(1)
        return pred


class SVM(LetterStatModel):
    # 已经可以正常使用
    train_ratio = 0.1

    def __init__(self, model_name):
        super(SVM, self).__init__(model_name)
        self.model = cv2.ml_SVM().create()

    def load(self, flag):
        fname = self.getSavePath(flag)
        self.model = cv2.ml.SVM_load(fname)
        savePickle("test_size_model.pkl", self.model)
        print "svm model loaded"

    def trainHog(self, samples, labels):
        # params = dict(kernel_type=cv2.ml.SVM_LINEAR,
        #               svm_type=cv2.ml.SVM_C_SVC,
        #               C=1)
        self.model.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels, 10,
                             cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_C),
                             cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_GAMMA),
                             cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_P),
                             cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_NU),
                             cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_COEF),
                             cv2.ml_SVM.getDefaultGridPtr(cv2.ml.SVM_DEGREE),
                             False
                             )
        print "svm train finished"


class MLP(LetterStatModel):
    # 已经可以正常使用
    def __init__(self, model_name):
        super(MLP, self).__init__(model_name)
        self.model = cv2.ml_ANN_MLP().create()

    def load(self, flag):
        fname = self.getSavePath(flag)
        self.model = cv2.ml.ANN_MLP_load(fname)
        print "mlp loaded"


    def trainHog(self, samples, labels):
        self.class_n = np.unique(labels).size
        sample_n, var_n = samples.shape
        new_labels = self.unroll_labels(labels).reshape(-1, self.class_n)

        layer_sizes = np.int32([var_n, 100, self.class_n])
        # self.model.create(layer_sizes)

        # CvANN_MLP_TrainParams::BACKPROP,0.001
        # params = dict(term_crit=(cv2.TERM_CRITERIA_COUNT, 300, 0.01),
        #               train_method=cv2.ml.ANN_MLP_BACKPROP,
        #               bp_dw_scale=0.001,
        #               bp_moment_scale=0.0)
        train_model = cv2.ml_ANN_MLP().create()
        train_model.setLayerSizes(layer_sizes)
        train_model.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001))  # 设置终止条件
        train_model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)  # 设置训练方式为反向传播
        train_model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
        train_model.setBackpropWeightScale(0.001)  # 设置反向传播中的一些参数
        train_model.setBackpropMomentumScale(0.0)  # 设置反向传播中的一些参数
        train_model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_labels))
        self.model = train_model
        print "mlp train finished"

    def predictHog(self, samples):
        _, resp = self.model.predict(samples, cv2.ml.ROW_SAMPLE)
        return resp.argmax(-1).astype(np.float32)


class Classifier:
    OCRMODEL = "ocr_model"   # 72752
    RTEMODEL = "rte_model"   # 1c63e
    KNNMODEL = "knn_model"
    BOOMODEL = 'boo_model'
    SVMMODEL = 'svm_model'   # 0aa03
    MLPMODEL = 'mlp_model'   # fe196

    def __init__(self, model_names=None, load_flag=""):
        self.model_names = model_names
        self._isTrained = False
        self.hogDesc = None
        self.classifier = None  # 分类器(knn, svm)
        # self.creatClassifier(model_name)  # 创建分类器
        self.label_dict = {}
        self.class_modes = {
            self.OCRMODEL: OCRModel,
            self.SVMMODEL: SVM,
            self.RTEMODEL: RTrees,
            self.MLPMODEL: MLP,
        }
        self.models = []
        if model_names:
            self.loadModelByNames(model_names)
        if load_flag:
            self.load(load_flag)

    def loadModelByNames(self, model_names):
        for model_name in model_names:
            model = self.class_modes.get(model_name)
            model_obj = model(model_name)
            self.models.append(model_obj)

    @staticmethod
    def load_XY_data(dirpath):
        """
        加载数据
        :param dirpath: 样本存放根目录(图片所在的目录文件夹名称即为分类)
        :return:
        """
        imgfiles = getTestFiles(dirpath)
        X = []
        Y = []
        for file in imgfiles:
            gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            X.append(gray)
            Y.append(int(file.split("/")[-2]))
        return np.array(X), np.array(Y)

    def train(self, imgs, labels):
        [model.train(imgs, labels) for model in self.models]

    def save(self, flag=""):
        [model.save(flag) for model in self.models]

    def load(self, flag=""):
        [model.load(flag) for model in self.models]


    def predict_one(self, img):
        predict_ = self.predicts([img, ])[0]
        return predict_[0], predict_[1]

    def predicts(self, imgs):
        more_results = np.array([model.predict(imgs) for model in self.models])
        from scipy import stats
        mode_res = stats.mode(more_results, axis=0, nan_policy='omit')
        labels = mode_res[0]
        score = mode_res[1]*1. / more_results.shape[0]
        labels_score = np.vstack((labels, score)).T
        labels_score_float32 = labels_score.astype(np.float32)
        results = []
        for label, score in labels_score_float32:
            results.append((int(label), score))
        return results


if __name__ == '__main__':
    imgfile = "/disk_d/poleImg/5d73e0f6538e4973ada74cffad7e6569/poleNum_1570877152669.jpg"
    gray = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    # ocr_model = OCR_MODEL("jj")
    # ocr_model.predict_one(gray)
    # svm_model = SVM_MODEL(Classifier.SVC_MODEL_NAME)
    # svm_model.predict_one(gray)
    sum_model = Classifier(model_names=[Classifier.RTEMODEL, Classifier.SVMMODEL, Classifier.MLPMODEL])
    # sum_model = Classifier(model_names=[Classifier.OCRMODEL])

    train_x, test_y = Classifier.load_XY_data("/disk_d/workspace/image/CPArea/bin128")
    train_x, test_x, train_y, test_y = train_test_split(train_x, test_y, test_size=.2, random_state=101)
    # train_x, test_x, train_y, test_y = classifier.load_XY_data("/disk_d/workspace/image/CPArea/10_bin128")
    # t1 = time.time()
    # sum_model.train(train_x, train_y)
    # sum_model.save("0_10")
    sum_model.load("0_10")
    t2 = time.time()
    for ts in test_x:
        predict_y = sum_model.predict_one(ts)
    # predict_y = sum_model.predict_one(test_x[0])
    t3 = time.time()
    score = np.where((predict_y[:,0] - test_y) == 0)[0].size*1. / test_y.size
    # print "训练耗时: {}; 预测耗时: {}; 准确率: {}".format(t2-t1, t3-t2, score)

