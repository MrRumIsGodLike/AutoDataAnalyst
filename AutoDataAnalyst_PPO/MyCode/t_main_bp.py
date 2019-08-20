# coding:utf-8
import tensorflow as tf
import pandas as pd
import time
import numpy as np
import warnings

from MyCode.t_Agent_1 import LSTM
from MyCode.EnvironmentManager import EnvironmentManager
from MyCode.configFile.MainConfigureFile import MainConfig
from MyCode.configFile.AgentConfigFile import AgentConfig
from MyCode.NNet import NNet
from MyCode.DataManager import DataManager


# 主函数
# 真实环境下训练,不使用预测网络,使用数据引导池
def t_main_bp(data_manager, plot_time_reward):
    data_manager = data_manager
    envManager = EnvironmentManager(data_manager)
    envManager.auto_create_multi_singleprocess_envs()
    plot_data = {"time": [], "rewards_max": [], "rewards_mean": [], "reward_min": []}
    for i in range(1):
        Env, params = envManager.next_environment()
        agent = LSTM(params)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            baseline_reward = 0
            a = [0, 1]
            # init_input:[[0 1]]
            init_input = np.array(a).reshape(1, 2)
            start_time = time.time()
            summary_writter = tf.summary.FileWriter("Mylog/test2", sess.graph)
            # 训练1000次
            for j in range(MainConfig.num_train):
                x, agr_params, action, _ = agent.getArgParams(sess, init_input)
                # 不使用神经网络
                rewards = Env.run(action)
                # 5:定义log写入流
                summarize(summary_writter, np.max(rewards), j, 'max_reward')
                summarize(summary_writter, np.mean(rewards), j, 'mean_reward')
                step_time = time.time()
                one_time = step_time - start_time
                plot_data["time"].append(one_time)
                plot_data["rewards_max"].append(np.max(rewards))
                plot_data["rewards_mean"].append(np.mean(rewards))
                plot_data["reward_min"].append(np.min(rewards))
                if j % 100 == 0:
                    plot = pd.DataFrame(data=plot_data)
                    plot.to_csv(plot_time_reward, index=False)
                # 记录下top 级的数据,等到合适的时候再放进模型训练；
                agent.check_topData(x, agr_params, rewards)
                if j == 0:
                    baseline_reward = np.mean(rewards)
                if (j + 1) % 10 == 0:
                    x, agr_params, rewards = agent.getInput()
                    print("if: algorithm rectify, rewards:", np.array(rewards).flatten())
                    loss, _ = agent.learn(True, sess, x, agr_params, rewards, baseline_reward, j)
                    print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=",
                          baseline_reward,
                          " loss=", loss, "\n")
                    summarize(summary_writter, loss, j, 'loss')

                else:
                    print("else: normal training, rewards:", rewards)
                    loss, _ = agent.learn(False, sess, x, agr_params, rewards, baseline_reward, j)
                    print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=",
                          baseline_reward,
                          " loss=", loss, "\n")
                    summarize(summary_writter, loss, j, 'loss')

                reward_c = np.mean(rewards)
                baseline_reward = baseline_reward * AgentConfig.dr + (1 - AgentConfig.dr) * reward_c

            # 存储300次 每次的实时reward和time
            plot = pd.DataFrame(data=plot_data)
            plot.to_csv(plot_time_reward, index=False)
    print("---------训练结束!----------")

# 添加tensorboard记录信息
def summarize(summary_writter, value, step, tag):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writter.add_summary(summary, step)
    summary_writter.flush()

# 使用image_segmentation数据集
data_manager = DataManager()
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
   # plot_time_reward = "../validate_time/params_data_agent(chen)/plot_time_data.csv"
    plot_time_reward = "../MyValidate_time/test2/plot_time_data.csv"
    t_main_bp(data_manager, plot_time_reward)
