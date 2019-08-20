# coding:utf-8
import tensorflow as tf
import numpy as np
import warnings
from MyCode.NNet import NNet
from MyCode.EnvironmentManager import EnvironmentManager
from MyCode.DataManager import DataManager
import time
import random

# 检测meta-feature对于数据集区分的作用
# 数据集1:image-segmentation / 数据集2:Crowdsourced / 数据集3:pr-handwritten /数据集4:optdigits
# 实验使用模型为RandomForestClassifier
# 不加meta-feature
def t_main_no_meta_diff():
    # 环境初始化
    nnet = NNet(5)
    data_manager_img = DataManager(6)
    data_manager_crowd = DataManager(12)
    data_manager_pr = DataManager(14)
    data_manager_opt = DataManager(9)

    envManagerImg = EnvironmentManager(data_manager_img)
    envManagerCrowd = EnvironmentManager(data_manager_crowd)
    envManagerPr = EnvironmentManager(data_manager_pr)
    envManagerOpt = EnvironmentManager(data_manager_opt)

    envManagerImg.auto_create_multi_singleprocess_envs()
    envManagerCrowd.auto_create_multi_singleprocess_envs()
    envManagerPr.auto_create_multi_singleprocess_envs()
    envManagerOpt.auto_create_multi_singleprocess_envs()

    env_img,_ = envManagerImg.next_environment()
    env_crowd,_ = envManagerCrowd.next_environment()
    env_pr,_ = envManagerPr.next_environment()
    env_opt,_ = envManagerOpt.next_environment()

    img_agr = np.loadtxt("../MyValidate_time/test-meta3/img_sample.csv", delimiter=",")
    crowd_agr = np.loadtxt("../MyValidate_time/test-meta3/crowdsourced_sample.csv", delimiter=",")
    pr_agr = np.loadtxt("../MyValidate_time/test-meta3/pr_sample.csv", delimiter=",")
    opt_agr = np.loadtxt("../MyValidate_time/test-meta3/optdigits_sample.csv",delimiter=",")

    # 训练预测网络(将四个数据集放在一起进行训练)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # summary_writter = tf.summary.FileWriter(path, sess.graph)
        # 1 2 3 4数据集训练预测网络 10steps
        for i in range(4):
            img_agr_train = img_agr[8 * i:8 * (i + 1), :5]
            img_agr_label = img_agr[8 * i:8 * (i + 1), 5]

            crowd_agr_train = crowd_agr[8 * i:8 * (i + 1), :5]
            crowd_agr_label = crowd_agr[8 * i:8 * (i + 1), 5]

            pr_agr_train = pr_agr[8 * i:8 * (i + 1), :5]
            pr_agr_label = pr_agr[8 * i:8 * (i + 1), 5]

            opt_agr_train = opt_agr[8 * i:8 * (i + 1), :5]
            opt_agr_label = opt_agr[8 * i:8 * (i + 1),5]

            nnet.store_transition(img_agr_train, img_agr_label)
            nnet.store_transition(crowd_agr_train, crowd_agr_label)
            nnet.store_transition(pr_agr_train, pr_agr_label)
            nnet.store_transition(opt_agr_train,opt_agr_label)
            nnet.train_net(sess, i)
        print('------预测网络训练结束(1 2 3 4数据集)------')

        # 定义随机动作用于对比实验
        # 对数据进行shuffle
        action_sample = np.loadtxt("../MyValidate_time/test-meta3/action_sample.csv", delimiter=",")
        agr_sample = np.loadtxt("../MyValidate_time/test-meta3/agr_sample.csv", delimiter=",")
        random.shuffle(action_sample)
        random.shuffle(agr_sample)

        # 记录所有预测网络reward
        reward_img_pre_total = []
        reward_crowd_pre_total = []
        reward_pr_pre_total = []
        reward_opt_pre_total = []

        # 定义训练轮数
        n_step = 2

        # 预测网络预测reward
        start_time = time.time()
        for i in range(n_step):
            img_ran_action = agr_sample[8 * i:8 * (i + 1),:]
            crowd_ran_action = agr_sample[8 * i:8 * (i + 1),:]
            pr_ran_action = agr_sample[8 * i:8 * (i + 1),:]
            opt_ran_action = agr_sample[8 * i:8 * (i + 1),:]

            reward_img_pre = nnet.get_reward(sess,img_ran_action)
            reward_crowd_pre = nnet.get_reward(sess,crowd_ran_action)
            reward_pr_pre = nnet.get_reward(sess,pr_ran_action)
            reward_opt_pre = nnet.get_reward(sess,opt_ran_action)

            reward_img_pre_total.append(reward_img_pre)
            reward_crowd_pre_total.append(reward_crowd_pre)
            reward_pr_pre_total.append(reward_pr_pre)
            reward_opt_pre_total.append(reward_opt_pre)
        step_time = time.time()
        pre_time = step_time - start_time
        print('预测网络耗时:',pre_time)
        print('------预测网络预测reward结束------')

        # 记录所有真实环境reward
        reward_img_true_total = []
        reward_crowd_true_total = []
        reward_pr_true_total = []
        reward_opt_true_total = []

        # 真实环境测试
        start_time = time.time()
        for i in range(n_step):
            reward_img_true = env_img.run(action_sample[8 * i:8 * (i + 1),:])
            reward_crowd_true = env_crowd.run(action_sample[8 * i:8 * (i + 1),:])
            reward_pr_true = env_pr.run(action_sample[8 * i:8 * (i + 1),:])
            reward_opt_true = env_opt.run(action_sample[8 * i:8 * (i + 1),:])

            reward_img_true_total.append(reward_img_true)
            reward_crowd_true_total.append(reward_crowd_true)
            reward_pr_true_total.append(reward_pr_true)
            reward_opt_true_total.append(reward_opt_true)
        step_time = time.time()
        true_time = step_time - start_time
        print('真实环境耗时:',true_time)
        print('------真实环境获得reward结束------')

        reward_img_pre_total = np.array(reward_img_pre_total).reshape(n_step,8)
        reward_img_true_total = np.array(reward_img_true_total).reshape(n_step,8)
        reward_crowd_pre_total = np.array(reward_crowd_pre_total).reshape(n_step,8)
        reward_crowd_true_total = np.array(reward_crowd_true_total).reshape(n_step,8)
        reward_pr_pre_total = np.array(reward_pr_pre_total).reshape(n_step,8)
        reward_pr_true_total = np.array(reward_pr_true_total).reshape(n_step,8)
        reward_opt_pre_total = np.array(reward_opt_pre_total).reshape(n_step,8)
        reward_opt_true_total = np.array(reward_opt_true_total).reshape(n_step,8)

        # 计算真实reward与预测网络预测的reward的距离
        dis_pre = np.square(np.mean(reward_img_pre_total - reward_img_true_total))
        dis_crowd = np.square(np.mean(reward_crowd_pre_total - reward_crowd_true_total))
        dis_pr = np.square(np.mean(reward_pr_pre_total - reward_pr_true_total))
        dis_opt = np.square(np.mean(reward_opt_pre_total - reward_opt_true_total))
        print('数据集1的误差为:',dis_pre)
        print('数据集2的误差为:',dis_crowd)
        print('数据集3的误差为:',dis_pr)
        print('数据集4的误差为:',dis_opt)
        print('------实验结束------')

# 添加tensorboard记录信息
def summarize(summary_writter, value, step, tag):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writter.add_summary(summary, step)
    summary_writter.flush()


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    t_main_no_meta_diff()
