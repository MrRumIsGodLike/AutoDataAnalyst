# coding: utf-8
import numpy as np
import sklearn.datasets as skdata
import MyCode.read_data as RD
import math
from math import log
from scipy import stats


# 元特征提取类
class MetaFeatureExtractor(object):
    def __init__(self, data_set_id=0):
        self.data_cv = None
        self.num_ins = None
        self.log_num_ins = None
        self.num_feature = None
        self.log_num_feature = None
        self.dimen = None
        self.log_dimen = None
        self.inv_dimen = None
        self.log_inv_dimen = None
        self.kurtosis_min = None
        self.kurtosis_max = None
        self.kurtosis_mean = None
        self.kurtosis_std = None
        self.skewness_min = None
        self.skewness_max = None
        self.skewness_mean = None
        self.skewness_std = None
        self.entropy = None
        self.read_data(data_set_id)
        self.extract_meta_feature()

    # 对数据集归一化(mean)
    def data_mean_norm(self, dataSet):
        for i in range(len(dataSet[1])):
            mean = np.mean(dataSet[:,i])
            std = np.std(dataSet[:,i])
            if mean == 0:
                dataSet[:,i] = 0
            else:
                if std == 0:
                    dataSet[:,i] = 0
                else:
                    dataSet[:,i] = (dataSet[:,i] - mean) / std
        return dataSet

    # 从指定文件读取数据(image_segmentation)
    def read_data(self, data_set_id):
        dataset = [RD.load_image_segmentation_data_set,
                   RD.load_wilt_data_set,
                   RD.load_Crowdsourced_Mapping_Data_Set,
                   RD.load_pr_handwritten_data_set,
                   RD.load_optdigits,
                   RD.load_letter_recognition_data_set,
                   RD.load_CTG_data_set]

        if data_set_id != 0 and data_set_id != 1 and data_set_id != 2 and data_set_id != 3:
            data, labels = dataset[data_set_id](return_X_y=True)

            # 对数据集进行归一化
            data = self.data_mean_norm(data)
            labels = (labels - np.mean(labels)) / np.std(labels)

            for _ in range(20):
                pi = np.random.permutation(len(data))
                data, labels = data[pi], labels[pi]

            self.data_cv = {'data_cv': data[:int(len(data) * 0.7)],
                            'labels_cv': labels[:int(len(data) * 0.7)],
                            'data_test': data[int(len(data) * 0.7):],
                            'labels_test': labels[int(len(data) * 0.7):]
                            }
            print("data_cv:", np.shape(self.data_cv['data_cv']), 'labels_cv:', np.shape(self.data_cv['labels_cv']))
            print("data_test:", np.shape(self.data_cv['data_test']), 'labels_test:',
                  np.shape(self.data_cv['labels_test']))
        else:
            train_x, train_y, test_x, test_y = dataset[data_set_id](return_X_y=True)

            # 对数据集进行归一化
            train_x = self.data_mean_norm(train_x)
            test_x = self.data_mean_norm(test_x)
            train_y = (train_y - np.mean(train_y)) / np.std(train_y)
            test_y = (test_y - np.mean(test_y)) / np.std(test_y)

            for _ in range(20):
                pi = np.random.permutation(len(train_x))
                train_x, train_y = train_x[pi], train_y[pi]
                pi = np.random.permutation(len(test_x))
                test_x, test_y = test_x[pi], test_y[pi]

            self.data_cv = {'data_cv': train_x,
                            'labels_cv': train_y,
                            'data_test': test_x,
                            'labels_test': test_y
                            }
            print("data_cv:", np.shape(self.data_cv['data_cv']), 'labels_cv:',
                  np.shape(self.data_cv['labels_cv']))
            print("data_test:", np.shape(self.data_cv['data_test']), 'labels_test:',
                  np.shape(self.data_cv['labels_test']))

    # 用于计算数据集的熵
    def calcShannonEnt(self, overall_data, overall_label):
        # 稀疏矩阵合并
        dataSet = np.column_stack((overall_data, overall_label))
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    # 用于提取数据集元特征
    def extract_meta_feature(self):
        overall_data = np.vstack((self.data_cv['data_cv'], self.data_cv['data_test']))
        overall_label = np.hstack((self.data_cv['labels_cv'], self.data_cv['labels_test']))
        print('数据及标签的维度:', overall_data.shape, overall_label.shape)

        # 开启元特征的提取
        # number of instances
        self.num_ins = len(overall_data)
        # log number of instances
        self.log_num_ins = math.log(self.num_ins)
        # number of features
        self.num_feature = len(overall_data[0])
        # log number of features
        self.log_num_feature = math.log(self.num_feature)
        # data set dimensionality
        self.dimen = self.num_ins * self.num_feature
        # log data set dimensionality
        self.log_dimen = math.log(self.dimen)
        # inverse data set dimensionality
        self.inv_dimen = 1 / self.dimen
        # log inverse data set dimensionality
        self.log_inv_dimen = math.log(self.inv_dimen)
        # Kurtosis min
        self.kurtosis_min = stats.kurtosis(overall_data).min()
        # kurtosis max
        self.kurtosis_max = stats.kurtosis(overall_data).max()
        # kurtosis mean
        self.kurtosis_mean = stats.kurtosis(overall_data).mean()
        # kurtosis std
        self.kurtosis_std = stats.kurtosis(overall_data).std()
        # skewness min
        self.skewness_min = stats.skew(overall_data).min()
        # skewness max
        self.skewness_max = stats.skew(overall_data).max()
        # skewness mean
        self.skewness_mean = stats.skew(overall_data).mean()
        # skewness std
        self.skewness_std = stats.skew(overall_data).std()
        # entropy
        self.entropy = self.calcShannonEnt(overall_data, overall_label)

        print('------元特征提取结束------')


if __name__ == "__main__":
    m1 = MetaFeatureExtractor(4)
