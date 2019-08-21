# -*- coding: utf-8 -*-
# from RL.AutoDataAnalyst_PPO.code.DataManager import DataManager
from MyCode.DataManager import DataManager
from RL.AutoDataAnalyst_PPO.code.t_main_1 import t_main_1
# from RL.AutoDataAnalyst_PPO.code.t_main_pre import t_main_pre
# from RL.AutoDataAnalyst_PPO.code.t_main_bp import t_main_bp
# from RL.AutoDataAnalyst_PPO.code.t_main_bp_pre import t_main_bp_pre
from MyCode.t_main_bp_pre import t_main_bp_pre
# from RL.AutoDataAnalyst_PPO.code.t_compared_test_gp import t_main_2
# from RL.AutoDataAnalyst_PPO.code.t_compared_test_tpe import t_main_3
from MyCode.t_compared_test_gp import t_main_2
from MyCode.t_compared_test_tpe import t_main_3

#from RL.AutoDataAnalyst_PPO.code.CMAES import CMAES
from MyCode.CMAES import CMAES

# from RL.AutoDataAnalyst_PPO.code.t_compared_test_rand import t_main_4
# from RL.AutoDataAnalyst_PPO.code.configFile.MainConfigureFile import MainConfig
from MyCode.configFile.MainConfigureFile import MainConfig

from chocolate import SQLiteConnection
import chocolate as choco
import os
import warnings

warnings.filterwarnings(action = 'ignore',category = DeprecationWarning)
for i in range(1):
    for j in range(MainConfig.dataset_size):
        print("第"+str(i)+"次实验:"+"加载第"+str(j)+"个数据集")
        data_manager = DataManager(data_set_index=j)
        # agent
        # print("第"+str(i)+"次实验:"+"加载第"+str(j)+"个数据集***agent***方法开始")
        # os.mkdir("../validate_time/params_data_agent(chen)/"+str(i)+"_agent_" + str(j))
        # os.mkdir("./data_dict/"+str(i)+"_agent_" + str(j))
        # path = str(i)+"_agent_" + str(j)
        # plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
        # t_main_1(data_manager, plot_time_reward)
        # print("第" + str(i) + "次实验:" + "加载第" + str(j) + "个数据集***agent***方法结束")
        # agent加预测和引导
        print("第" + str(i) + "次实验:" + "加载第" + str(j) + "个数据集***agent—RPR***方法开始")
        os.mkdir("../validate_time/params_data_agent(chen)/"+str(i)+"_rpr_" + str(j))
        os.mkdir("./data_dict/"+str(i)+"_rpr_" + str(j))
        path = str(i)+"_rpr_" + str(j)
        plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
        t_main_bp_pre(data_manager, plot_time_reward)
        print("第" + str(i) + "次实验:" + "加载第" + str(j) + "个数据集***agent-RPR***方法结束")
        # TPE方法
        print("第" + str(i) + "次实验:" + "加载第" + str(j) + "个数据集***TPE***方法开始")
        os.mkdir("../validate_time/params_data_tpe/"+str(i)+"_tpe_" + str(j))
        os.mkdir("./data_dict/"+str(i)+"_tpe_" + str(j))
        path = str(i)+"_tpe_" + str(j)
        file_name = "../validate_time/params_data_tpe/" + path + "/logs.txt"
        data_file_name = "../validate_time/params_data_tpe/" + path + "/data.txt"
        plot_data_path = "../validate_time/params_data_tpe/" + path + "/plot_data.csv"
        data_dict_file = "./data_dict/" + path + "/t_tpe.csv"
        t_main_3(data_manager, file_name, data_file_name, plot_data_path, data_dict_file)
        print("第" + str(i) + "次实验:" + "加载第" + str(j) + "个数据集***TPE***方法结束")
        # CMAES 方法
        print("第" + str(i) + "次实验:" + "加载第" + str(j) + "个数据集***CMAES***方法开始")
        n=1
        os.mkdir("../validate_time/params_data_cmaes/" + str(i) + "_cmaes_" + str(j))
        path = str(i) + "_cmaes_" + str(j)
        file_name = "../validate_time/params_data_cmaes/" + path + "/data.csv"
        url_path = "sqlite:///" + "../validate_time/params_data_cmaes/" + path + "/mnistdb.db"
        conn = choco.SQLiteConnection(url=url_path)
        CMAES(data_manager, n, file_name, conn)
        print("第" + str(i) + "次实验:" + "加载第" + str(j) + "个数据集***CMAES***方法结束")
        # # agent加引导  手写数字
        # os.mkdir("../validate_time/params_data_agent(chen)/digits_bp_" + str(i))
        # os.mkdir("./data_dict/digits_bp_" + str(i))
        # path = "digits_bp_" + str(i)
        # plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
        # t_main_bp(data_manager_digits, plot_time_reward)
        # # agent加预测 手写数字
        # os.mkdir("../validate_time/params_data_agent(chen)/digits_pre_" + str(i))
        # os.mkdir("./data_dict/digits_pre_" + str(i))
        # path = "digits_pre_" + str(i)
        # plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
        # t_main_pre(data_manager_digits, plot_time_reward)
        # 基于贝叶斯优化 手写数字
        # os.mkdir("../validate_time/params_data_gp/digits_gp_" + str(i))
        # os.mkdir("./data_dict/digits_gp_" + str(i))
        # path = "digits_gp_" + str(i)
        # file_name = "../validate_time/params_data_gp/" + path + "/logs.txt"
        # data_file_name = "../validate_time/params_data_gp/" + path + "/data.txt"
        # plot_data_path = "../validate_time/params_data_gp/" + path + "/plot_data.csv"
        # data_dict_file = "./data_dict/" + path + "/t_gp.csv"
        # t_main_2(data_manager_digits, file_name, data_file_name, plot_data_path, data_dict_file)
        # 基于rand随机 手写字母
        # os.mkdir("../validate_time/params_data_rand/litter_rand_" + str(i))
        # os.mkdir("./data_dict/litter_rand_" + str(i))
        # path = "litter_rand_" + str(i)
        # file_name = "../validate_time/params_data_rand/" + path + "/logs.txt"
        # data_file_name = "../validate_time/params_data_rand/" + path + "/data.txt"
        # plot_data_path = "../validate_time/params_data_rand/" + path + "/plot_data.csv"
        # data_dict_file = "./data_dict/" + path + "/t_rand.csv"
        # t_main_4(data_manager_letter, file_name, data_file_name, plot_data_path, data_dict_file)

