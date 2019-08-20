# coding:utf-8
import tensorflow as tf
import pandas as pd
import time
import numpy as np
import warnings
from MyCode.DataManager import DataManager
from MyCode.t_Agent_1 import LSTM
from MyCode.EnvironmentManager import EnvironmentManager
from MyCode.configFile.MainConfigureFile import MainConfig
from MyCode.configFile.AgentConfigFile import AgentConfig
from MyCode.NNet import NNet
from MyCode.MetaFeatureExtractor import MetaFeatureExtractor


# 主函数:
# 使用预测网络 不使用数据引导池
def t_main_pre(data_manager, plot_time_reward):
    data_manager = data_manager
    m = MetaFeatureExtractor(4)
    meta_vec = [m.num_ins, m.log_num_ins, m.num_feature, m.log_num_feature,
                m.dimen, m.log_dimen, m.inv_dimen, m.log_inv_dimen, m.kurtosis_min,
                m.kurtosis_max, m.kurtosis_mean, m.kurtosis_std, m.skewness_min,
                m.skewness_max, m.skewness_mean, m.skewness_std, m.entropy]
    meta_vec = [meta_vec] * 8
    meta_vec = np.array(meta_vec)
    envManager = EnvironmentManager(data_manager)
    envManager.auto_create_multi_singleprocess_envs()
    plot_data = {"time": [], "rewards_max": [], "rewards_mean": [], "reward_min": []}
    for i in range(1):
        # 当前Env为随机森林,params表示该算法超参数预选值维度 如[12 10 21 21 9]
        Env, params = envManager.next_environment()
        agent = LSTM(params)
        nnet = NNet(len(params) + 17)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            baseline_reward = 0
            a = [0, 1]
            init_input = np.array(a).reshape(1, 2)
            start_time = time.time()
            summary_writter = tf.summary.FileWriter("Mylog/test-meta2", sess.graph)
            # 训练1000次
            for j in range(5):
                # init_input:[[0 1]]
                # action为选取到的动作,agr_params为对应的正态分布下的值
                x, agr_params, action, _ = agent.getArgParams(sess, init_input)
                data1 = pd.DataFrame(agr_params)
                data2 = pd.DataFrame(action)
                data1.to_csv('../MyValidate_time/test-meta3/agr_sample.csv', index=False, header=False, mode='a')
                data2.to_csv('../MyValidate_time/test-meta3/action_sample.csv', index=False, header=False, mode='a')

                # print('本次得到的action为:',action)
                # stack_data = np.hstack((agr_params, meta_vec))
                # # 使用神经网络
                # # 前25轮训练预测网络
                # if j <= MainConfig.t1:
                #     rewards = Env.run(action)
                #     # 将样本加入到文件中
                #     data1 = pd.DataFrame(np.c_[agr_params, rewards])
                #     # data1.to_csv('../MyValidate_time/test-meta3/pr_sample.csv', index=False, header=False, mode='a')
                #     # data1.to_csv('../MyValidate_time/test-meta3/img_sample.csv', index=False, header=False, mode='a')
                #     # data1.to_csv('../MyValidate_time/test-meta3/crowdsourced_sample.csv', index=False, header=False,
                #     #             mode='a')
                #     data1.to_csv('../MyValidate_time/test-meta3/optdigits_sample.csv', index=False, header=False,mode='a')
                #     summarize(summary_writter, np.max(rewards), j, 'max_reward')
                #     summarize(summary_writter, np.mean(rewards), j, 'mean_reward')
                #     step_time = time.time()
                #     one_time = step_time - start_time
                #     plot_data["time"].append(one_time)
                #     plot_data["rewards_max"].append(np.max(rewards))
                #     plot_data["rewards_mean"].append(np.mean(rewards))
                #     plot_data["reward_min"].append(np.min(rewards))
                #     nnet.store_transition(stack_data, rewards)
                #     nnet.train_net(sess, j)
                #     nnet.train_net(sess, j)
                # if j > MainConfig.t1 and j < MainConfig.t2:
                #     rewards = nnet.get_reward(sess, stack_data)
                #     rewards = np.array(rewards).reshape(AgentConfig.batch_size)
                #     summarize(summary_writter, np.max(rewards), j, 'max_reward')
                #     summarize(summary_writter, np.mean(rewards), j, 'mean_reward')
                # if j >= MainConfig.t2:
                #     rewards = Env.run(action)
                #     step_time = time.time()
                #     one_time = step_time - start_time
                #     plot_data["time"].append(one_time)
                #     plot_data["rewards_max"].append(np.max(rewards))
                #     plot_data["rewards_mean"].append(np.mean(rewards))
                #     plot_data["reward_min"].append(np.min(rewards))
                #     summarize(summary_writter, np.max(rewards), j, 'max_reward')
                #     summarize(summary_writter, np.mean(rewards), j, 'mean_reward')
                # if j % 100 == 0:
                #     plot = pd.DataFrame(data=plot_data)
                #     plot.to_csv(plot_time_reward, index=False)
                # if j == 0:
                #     baseline_reward = np.mean(rewards)
                # print("else: normal training, rewards:", rewards)
                # loss, ratio = agent.learn(False, sess, x, agr_params, rewards, baseline_reward, j)
                # print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=", baseline_reward,
                #       " loss=", loss, "\n")
                # summarize(summary_writter, loss, j, 'loss')
                # summarize(summary_writter, ratio, j, 'ratio')
                # reward_c = np.mean(rewards)
                # baseline_reward = baseline_reward * AgentConfig.dr + (1 - AgentConfig.dr) * reward_c
            # 存储300次 每次的实时reward和time
            # plot = pd.DataFrame(data=plot_data)
            # plot.to_csv(plot_time_reward, index=False)
    print("---------训练结束!----------")


# 添加tensorboard记录信息
def summarize(summary_writter, value, step, tag):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writter.add_summary(summary, step)
    summary_writter.flush()


# 默认image_segmentation数据集
# data_manager = DataManager(6)

# data_manager = DataManager(14)
# data_manager = DataManager(12)
data_manager = DataManager(9) # RD.load_optdigits
# data_manager = DataManager(17) # RD.load_letter_recognition_data_set
# data_manager = DataManager(5) # RD.load_CTG_data_set
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    # plot_time_reward = "../validate_time/params_data_agent(chen)/plot_time_data.csv"
    plot_time_reward = "../MyValidate_time/test-meta2/plot_time_data.csv"
    t_main_pre(data_manager, plot_time_reward)
