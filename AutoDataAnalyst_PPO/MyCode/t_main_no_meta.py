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
def t_main_no_meta():
    nnet = NNet(5)
    img_agr = np.loadtxt("../MyValidate_time/test-meta3/img_sample.csv", delimiter=",")
    crowd_agr = np.loadtxt("../MyValidate_time/test-meta3/crowdsourced_sample.csv", delimiter=",")
    pr_agr = np.loadtxt("../MyValidate_time/test-meta3/pr_sample.csv", delimiter=",")
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    path = "Mylog/test-meta3-no/" + TIMESTAMP

    # 训练预测网络(将两个数据集放在一起进行训练)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # summary_writter = tf.summary.FileWriter(path, sess.graph)
        # 1和2数据集训练预测网络
        for i in range(10):
            img_agr_train = img_agr[8 * i:8 * (i + 1), :5]
            img_agr_label = img_agr[8 * i:8 * (i + 1), 5]

            crowd_agr_train = crowd_agr[8 * i:8 * (i + 1), :5]
            crowd_agr_label = crowd_agr[8 * i:8 * (i + 1), 5]

            nnet.store_transition(img_agr_train, img_agr_label)
            nnet.store_transition(crowd_agr_train, crowd_agr_label)
            nnet.train_net(sess, i)
        print('------预测网络训练结束(1和2数据集)------')

        # 3数据集训练预测网络
        for k in range(10):
            pr_agr_train = pr_agr[8 * k:8 * (k + 1), :5]
            pr_agr_label = pr_agr[8 * k:8 * (k + 1), 5]
            nnet.pre_train_net(sess,k,pr_agr_train,pr_agr_label)
        print('------预测网络训练结束(3数据集)')

        # 预测数据
        for j in range(90):
            pr_agr_train = pr_agr[8 * j:8 * (j + 1), :5]
            pr_agr_label = pr_agr[8 * j:8 * (j + 1), 5]
            # rewards = nnet.get_reward(sess, crowd_agr_train)

            reward = nnet.pre_get_reward(sess, j, pr_agr_train, pr_agr_label)

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
    t_main_no_meta()
