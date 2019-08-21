# coding:utf-8
import tensorflow as tf
import pandas as pd
import time
import numpy as np
import warnings
# from RL.AutoDataAnalyst_PPO.code.DataManager import DataManager
# from RL.AutoDataAnalyst_PPO.code.t_Agent_1 import LSTM
# from RL.AutoDataAnalyst_PPO.code.EnvironmentManager import EnvironmentManager
# from RL.AutoDataAnalyst_PPO.code.configFile.MainConfigureFile import MainConfig
# from RL.AutoDataAnalyst_PPO.code.configFile.AgentConfigFile import AgentConfig
# from RL.AutoDataAnalyst_PPO.code.NNet import NNet
from MyCode.DataManager import DataManager
from MyCode.t_Agent_1 import LSTM
from MyCode.EnvironmentManager import EnvironmentManager
from MyCode.configFile.MainConfigureFile import MainConfig
from MyCode.configFile.AgentConfigFile import AgentConfig
from MyCode.NNet import NNet


def t_main_bp_pre_test(data_manager,plot_time_reward):
    data_manager = data_manager
    envManager = EnvironmentManager(data_manager)
    envManager.auto_create_multi_singleprocess_envs()
    plot_data = {"time": [], "rewards_max": [], "rewards_mean": [], "reward_min": []}
    for i in range(1):
        Env, params = envManager.next_environment()
        agent = LSTM(params)
        nnet = NNet(len(params))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            baseline_reward = 0
            a = [0, 1]
            init_input = np.array(a).reshape(1, 2)
            start_time = time.time()
            summary_writter_reward = tf.summary.FileWriter("log/reward", sess.graph)
            summary_writter_reward_real = tf.summary.FileWriter("log/reward_real", sess.graph)
            summary_writter_baseline = tf.summary.FileWriter("log/baseline", sess.graph)
            summary_writter_baseline_real = tf.summary.FileWriter("log/baseline_real", sess.graph)
            summary_writter_err = tf.summary.FileWriter("log/err", sess.graph)
            for j in range(MainConfig.num_train):
                x, agr_params, action, _ = agent.getArgParams(sess, init_input)
                # 使用神经网络
                if j <= 25:
                    rewards = Env.run(action)
                    nnet.store_transition(agr_params, rewards)
                    nnet.train_net(sess, j)
                    nnet.train_net(sess, j)
                    rewards_real = rewards
                if j > 25 :
                    rewards = nnet.get_reward(sess, agr_params)
                    rewards = np.array(rewards).reshape(AgentConfig.batch_size)
                    rewards_real = Env.run(action)
                summarize(summary_writter_reward, np.mean(rewards), j, "reward")
                summarize(summary_writter_reward_real, np.mean(rewards_real), j, "reward")
                # if j >= 150:
                #     rewards = Env.run(action)
                # 记录下top 级的数据,等到合适的时候再放进模型训练；
                # if j <= 25 or j >= 150:
                #     agent.check_topData(x, agr_params, rewards)
                if j == 0:
                    baseline_reward = np.mean(rewards)
                    baseline_reward_real = np.mean(rewards_real)
                # # 每十步 讲使用引导数据池 即最好的reward的超参数值   用于减少方差
                # if (j + 1) % 10 == 0:
                #     x, agr_params, rewards = agent.getInput()
                #     print("if: algorithm rectify, rewards:", np.array(rewards).flatten())
                #     loss, ratio = agent.learn(True, sess, x, agr_params, rewards, baseline_reward, j)
                #     print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=",
                #           baseline_reward,
                #           " loss=", loss, "\n")
                # else:
                print("else: normal training, rewards:", rewards)
                loss, ratio = agent.learn(False, sess, x, agr_params, rewards, baseline_reward, j)
                print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=",
                      baseline_reward,
                      " loss=", loss, "\n")
                reward_c = np.mean(rewards)
                reward_c_real = np.mean(rewards_real)
                baseline_reward = baseline_reward * AgentConfig.dr + (1-AgentConfig.dr) * reward_c
                baseline_reward_real = baseline_reward_real * AgentConfig.dr + (1 - AgentConfig.dr) * reward_c_real
                err_all = (reward_c - np.mean(baseline_reward)) - (reward_c_real - np.mean(baseline_reward_real))
                summarize(summary_writter_baseline, np.mean(baseline_reward), j, "baseline")
                summarize(summary_writter_baseline_real, np.mean(baseline_reward_real), j, "baseline")
                summarize(summary_writter_err, np.mean(err_all), j, "err")
            #存储300次 每次的实时reward和time
            plot = pd.DataFrame(data=plot_data)
            plot.to_csv(plot_time_reward, index=False)
    print("---------训练结束!----------")
def summarize(summary_writter,value,step,tag):
    summary=tf.Summary()
    summary.value.add(tag=tag,simple_value=value)
    summary_writter.add_summary(summary,step)
    summary_writter.flush()
data_manager=DataManager()
if __name__ == '__main__':
    warnings.filterwarnings(action = 'ignore',category = DeprecationWarning)
    plot_time_reward = "../validate_time/params_data_agent(chen)/plot_time_data.csv"
    t_main_bp_pre_test(data_manager,plot_time_reward)
