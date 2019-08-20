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


def t_main_bp_pre(data_manager,plot_time_reward):
    data_manager = data_manager
    envManager = EnvironmentManager(data_manager)
    envManager.auto_create_multi_singleprocess_envs()
    plot_data = {"time": [], "rewards_max": [], "rewards_mean": [], "reward_min": []}
    real_data = {"reward": [], "action": []}
    pre_data = {"reward": [], "action": []}
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
            for i in range(1):
                nnet._init_net(sess,len(params))
                for j in range(MainConfig.num_train):
                    x, agr_params, action, _ = agent.getArgParams(sess, init_input)
                    np.set_printoptions(suppress=True)

                    # 使用神经网络
                    rewards = Env.run(action)
                    for ii in range(8):
                        real_data["action"].append(agr_params[ii])
                        real_data["reward"].append(rewards[ii])
                    np.savetxt("../validate_time/params_data_agent(chen)/real_data_action.csv", real_data["action"],
                               delimiter=',')
                    np.savetxt("../validate_time/params_data_agent(chen)/real_data_reward.csv", real_data["reward"],
                               delimiter=',')
                    step_time = time.time()
                    one_time = step_time - start_time
                    plot_data["time"].append(one_time)
                    plot_data["rewards_max"].append(np.max(rewards))
                    plot_data["rewards_mean"].append(np.mean(rewards))
                    plot_data["reward_min"].append(np.min(rewards))
                    nnet.store_transition(agr_params, rewards)
                    nnet.train_net(sess, j)
                    nnet.train_net(sess, j)
                    if j % MainConfig.num_train - 1 == 0:
                        plot = pd.DataFrame(data=plot_data)
                        plot.to_csv(plot_time_reward, index=False)
                    agent.check_topData(x, agr_params, rewards)
                    if j == 0:
                        baseline_reward = np.mean(rewards)
                    # 每十步 讲使用引导数据池 即最好的reward的超参数值   用于减少方差
                    if (j + 1) % 10 == 0:
                        x, agr_params, rewards = agent.getInput()
                        print("if: algorithm rectify, rewards:", np.array(rewards).flatten())
                        loss, ratio = agent.learn(True, sess, x, agr_params, rewards, baseline_reward, j)
                        print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=",
                              baseline_reward,
                              " loss=", loss, "\n")
                    else:
                        print("else: normal training, rewards:", rewards)
                        loss, ratio = agent.learn(False, sess, x, agr_params, rewards, baseline_reward, j)
                        print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=",
                              baseline_reward,
                              " loss=", loss, "\n")
                    reward_c = np.mean(rewards)
                    baseline_reward = baseline_reward * AgentConfig.dr + (1-AgentConfig.dr) * reward_c
                for j in range(MainConfig.pre_num):
                    x, agr_params, action, _ = agent.getArgParams(sess, init_input)
                    np.set_printoptions(suppress=True)
                    # 使用model
                    rewards = nnet.get_reward(sess, agr_params)
                    for ii in range(8):
                        pre_data["action"].append(agr_params[ii])
                        pre_data["reward"].append(rewards[ii])
                    # if j % (MainConfig.pre_num-1) == 0:
                    np.savetxt("../validate_time/params_data_agent(chen)/pre_data_action.csv", pre_data["action"],
                               delimiter=',')
                    np.savetxt("../validate_time/params_data_agent(chen)/pre_data_reward.csv", pre_data["reward"],
                               delimiter=',')
                    rewards = np.array(rewards).reshape(AgentConfig.batch_size)
                    loss, ratio = agent.learn(False, sess, x, agr_params, rewards, baseline_reward, j)
                    reward_c = np.mean(rewards)
                    baseline_reward = baseline_reward * AgentConfig.dr + (1 - AgentConfig.dr) * reward_c

            #存储300次 每次的实时reward和time
            plot = pd.DataFrame(data=plot_data)
            plot.to_csv(plot_time_reward, index=False)
    print("---------训练结束!----------")

data_manager = DataManager(17)
if __name__ == '__main__':
    warnings.filterwarnings(action = 'ignore',category = DeprecationWarning)
    plot_time_reward = "../MyValidate_time/test4/plot_time_data.csv"

    t_main_bp_pre(data_manager,plot_time_reward)
