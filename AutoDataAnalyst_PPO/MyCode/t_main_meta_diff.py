# coding:utf-8
import tensorflow as tf
import numpy as np
import warnings
from MyCode.NNet import NNet
from MyCode.MetaFeatureExtractor import MetaFeatureExtractor
from MyCode.EnvironmentManager import EnvironmentManager
from MyCode.DataManager import DataManager
import time
import random

# 检测meta-feature对于数据集区分的作用
# 数据集1:image-segmentation / 数据集2:Crowdsourced / 数据集3:pr-handwritten /数据集4:optdigits
# 实验使用模型为RandomForestClassifier
# 使用meta-feature
def t_main_meta_diff():
    # 环境初始化
    nnet = NNet(22)
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

    # 获取optdigits数据集的元特征
    m3 = MetaFeatureExtractor(4)
    meta_vec_opt = [m3.num_ins, m3.log_num_ins, m3.num_feature, m3.log_num_feature,
                    m3.dimen, m3.log_dimen, m3.inv_dimen, m3.log_inv_dimen, m3.kurtosis_min,
                    m3.kurtosis_max, m3.kurtosis_mean, m3.kurtosis_std, m3.skewness_min,
                    m3.skewness_max, m3.skewness_mean, m3.skewness_std, m3.entropy]
    # 使用高斯分布处理元特征
    meta_vec_opt = (meta_vec_opt - np.mean(meta_vec_opt)) / np.std(meta_vec_opt)
    meta_vec_opt = [meta_vec_opt] * 8
    meta_vec_opt = np.array(meta_vec_opt)

    img_agr = np.loadtxt("../MyValidate_time/test-meta3/img_sample.csv", delimiter=",")
    crowd_agr = np.loadtxt("../MyValidate_time/test-meta3/crowdsourced_sample.csv", delimiter=",")
    pr_agr = np.loadtxt("../MyValidate_time/test-meta3/pr_sample.csv", delimiter=",")
    opt_agr = np.loadtxt("../MyValidate_time/test-meta3/optdigits_sample.csv",delimiter=",")

    # 训练预测网络(将4个数据集放在一起进行训练)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # summary_writter = tf.summary.FileWriter(path, sess.graph)
        # 1 2 3 4数据集训练预测网络 10steps
        for i in range(4):
            img_agr_train = np.hstack((img_agr[8 * i:8 * (i + 1), :5], meta_vec_img))
            img_agr_label = img_agr[8 * i:8 * (i + 1), 5]

            crowd_agr_train = np.hstack((crowd_agr[8 * i:8 * (i + 1), :5], meta_vec_crowd))
            crowd_agr_label = crowd_agr[8 * i:8 * (i + 1), 5]

            pr_agr_train = np.hstack((pr_agr[8 * i:8 * (i + 1), :5], meta_vec_pr))
            pr_agr_label = pr_agr[8 * i:8 * (i + 1), 5]

            opt_agr_train = np.hstack((opt_agr[8 * i:8 * (i + 1), :5],meta_vec_opt))
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
            img_ran_action = np.hstack((agr_sample[8 * i:8 * (i + 1),:],meta_vec_img))
            crowd_ran_action = np.hstack((agr_sample[8 * i:8 * (i + 1),:],meta_vec_crowd))
            pr_ran_action = np.hstack((agr_sample[8 * i:8 * (i + 1),:],meta_vec_pr))
            opt_ran_action = np.hstack((agr_sample[8 * i:8 * (i + 1),:],meta_vec_opt))

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
    t_main_meta_diff()
