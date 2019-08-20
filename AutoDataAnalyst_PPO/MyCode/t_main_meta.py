# coding:utf-8
import tensorflow as tf
import pandas as pd
import time
import numpy as np
import warnings
from MyCode.DataManager import DataManager
from MyCode.NNet import NNet
from MyCode.MetaFeatureExtractor import MetaFeatureExtractor
from datetime import datetime


# 检测meta-features对预测网络的作用
# 学习Crowdsourced_Mapping数据和image-segmentation数据,预测pr-handwritten数据
# 当前测试模型:RandomForestClassifier
def t_main_meta():
    nnet = NNet(22)
    # 获取image-segmentation数据集的元特征
    m = MetaFeatureExtractor(0)
    meta_vec_img = [m.num_ins, m.log_num_ins, m.num_feature, m.log_num_feature,
                    m.dimen, m.log_dimen, m.inv_dimen, m.log_inv_dimen, m.kurtosis_min,
                    m.kurtosis_max, m.kurtosis_mean, m.kurtosis_std, m.skewness_min,
                    m.skewness_max, m.skewness_mean, m.skewness_std, m.entropy]
    # 使用高斯分布处理元特征
    meta_vec_img = (meta_vec_img - np.mean(meta_vec_img)) / np.std(meta_vec_img)
    meta_vec_img = [meta_vec_img] * 8
    meta_vec_img = np.array(meta_vec_img)

    # 获取Crowdsourced数据集的元特征
    m1 = MetaFeatureExtractor(2)
    meta_vec_crowd = [m1.num_ins, m1.log_num_ins, m1.num_feature, m1.log_num_feature,
                      m1.dimen, m1.log_dimen, m1.inv_dimen, m1.log_inv_dimen, m1.kurtosis_min,
                      m1.kurtosis_max, m1.kurtosis_mean, m1.kurtosis_std, m1.skewness_min,
                      m1.skewness_max, m1.skewness_mean, m1.skewness_std, m1.entropy]
    # 使用高斯分布处理元特征
    meta_vec_crowd = (meta_vec_crowd - np.mean(meta_vec_crowd)) / np.std(meta_vec_crowd)
    meta_vec_crowd = [meta_vec_crowd] * 8
    meta_vec_crowd = np.array(meta_vec_crowd)

    # 获取pr-handwritten数据集的元特征
    m2 = MetaFeatureExtractor(3)
    meta_vec_pr = [m2.num_ins, m2.log_num_ins, m2.num_feature, m2.log_num_feature,
                   m2.dimen, m2.log_dimen, m2.inv_dimen, m2.log_inv_dimen, m2.kurtosis_min,
                   m2.kurtosis_max, m2.kurtosis_mean, m2.kurtosis_std, m2.skewness_min,
                   m2.skewness_max, m2.skewness_mean, m2.skewness_std, m2.entropy]
    # 使用高斯分布处理元特征
    meta_vec_pr = (meta_vec_pr - np.mean(meta_vec_pr)) / np.std(meta_vec_pr)
    meta_vec_pr = [meta_vec_pr] * 8
    meta_vec_pr = np.array(meta_vec_pr)

    img_agr = np.loadtxt("../MyValidate_time/test-meta3/img_sample.csv", delimiter=",")
    crowd_agr = np.loadtxt("../MyValidate_time/test-meta3/crowdsourced_sample.csv", delimiter=",")
    pr_agr = np.loadtxt("../MyValidate_time/test-meta3/pr_sample.csv", delimiter=",")
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    path = "Mylog/test-meta3/" + TIMESTAMP

    # 训练预测网络(将两个数据集放在一起进行训练)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # summary_writter = tf.summary.FileWriter(path, sess.graph)
        # 1和2数据集训练预测网络
        for i in range(10):
            img_agr_train = np.hstack((img_agr[8 * i:8 * (i + 1), :5], meta_vec_img))
            img_agr_label = img_agr[8 * i:8 * (i + 1), 5]

            crowd_agr_train = np.hstack((crowd_agr[8 * i:8 * (i + 1), :5], meta_vec_crowd))
            crowd_agr_label = crowd_agr[8 * i:8 * (i + 1), 5]

            nnet.store_transition(img_agr_train, img_agr_label)
            nnet.store_transition(crowd_agr_train, crowd_agr_label)
            nnet.train_net(sess, i)
        print('------预测网络训练结束(1和2数据集)------')

        # 3数据集训练预测网络
        for k in range(10):
            pr_agr_train = np.hstack((pr_agr[8 * k:8 * (k + 1), :5], meta_vec_pr))
            pr_agr_label = pr_agr[8 * k:8 * (k + 1), 5]
            nnet.pre_train_net(sess,k,pr_agr_train,pr_agr_label)
        print('------预测网络训练结束(3数据集)')

        # 预测数据
        for j in range(90):
            pr_agr_train = np.hstack((pr_agr[8 * j:8 * (j + 1), :5], meta_vec_pr))
            pr_agr_label = pr_agr[8 * j:8 * (j + 1), 5]
            # rewards = nnet.get_reward(sess, crowd_agr_train)
            reward = nnet.pre_get_reward(sess,j,pr_agr_train,pr_agr_label)

            # loss = np.square(np.mean(rewards) - np.mean(crowd_agr_label))
            # summarize(summary_writter, np.max(rewards), j, 'max_reward')
            # summarize(summary_writter, np.mean(rewards), j, 'mean_reward')
            # summarize(summary_writter, np.max(pr_agr_label), j, 'max_label')
            # summarize(summary_writter, np.mean(pr_agr_label), j, 'mean_label')
            # summarize(summary_writter,loss,j,'loss')
            # print('本次的预测奖励值为:',rewards)
            # print('本次的真实奖励值为:',pr_agr_label)
        print('------预测结束------')


# 添加tensorboard记录信息
def summarize(summary_writter, value, step, tag):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writter.add_summary(summary, step)
    summary_writter.flush()


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    t_main_meta()
