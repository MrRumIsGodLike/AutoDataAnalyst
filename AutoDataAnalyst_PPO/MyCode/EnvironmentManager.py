# coding: utf-8
import numpy as np
import pandas as pd

# from RL.AutoDataAnalyst_PPO.code.configFile.ClassificationAlgorithmConfigureFile import ClassificationAlgorithmConfigure
# from RL.AutoDataAnalyst_PPO.code.Trainer import Trainer
# from RL.AutoDataAnalyst_PPO.code.configFile.AgentConfigFile import AgentConfig
from MyCode.configFile.ClassificationAlgorithmConfigureFile import ClassificationAlgorithmConfigure
from MyCode.Trainer import Trainer
from MyCode.configFile.AgentConfigFile import AgentConfig

# 训练环境类：单进程版
class SingleProcessEnvironment(object):
    def __init__(self, data_manager, c_config, c_i):
        self.data_manager = data_manager
        self.c_config = c_config
        self.c_i = c_i
        self.state_dimensionality_list = np.reshape(np.array(self.c_config.config_list[c_i], dtype=np.int), [-1])
        self.hot_method = {"paras": [], "rewards": []}
        self.data_dict_file = "./data_dict/data.csv"

    # 实现并行的模型计算
    def processor(self, actions):
        # 根据选择的动作，匹配相关算法设置，构建算法训练体
        trainer = Trainer(actions, self.c_config, self.c_i, self.data_manager)
        # 交叉验证
        accuracy = trainer.run_CV()
        return accuracy

    # 获得param列表
    def getparams(self,actions_list):
        params = []
        queue_length = len(actions_list)
        for loc in range (queue_length):
            param = []
            # 有多少个超参数
            key_value = self.c_config.methods_dict[self.c_i][1:]
            assert len(actions_list[loc]) == len(key_value), "Trainer.getparams: 数据维度应该一样！"
            for i in range(len(key_value)):
                a = key_value[i][1][actions_list[loc][i]]
                if a == False:
                    a = 0
                if a == True:
                    a = 1
                param.append(a)
            param = np.array(param).reshape(len(key_value),1)
            params.append(param)
        params = np.array(params).reshape(AgentConfig.batch_size,len(key_value))
        return params

    # action_list选择那个参数预选值的列表
    def run(self, actions_list):
        rewards = []
        queue_length = len(actions_list)
        for loc in range(queue_length):
            if np.any(str(actions_list[loc]) in self.hot_method["paras"]):
                index = self.hot_method["paras"].index(str(actions_list[loc]))
                rewards.append(self.hot_method["rewards"][index])
            else:
                reward = self.processor(actions_list[loc])
                rewards.append(reward)
                self.hot_method["paras"].append(str(actions_list[loc]))
                self.hot_method["rewards"].append(reward)

        return rewards

    def get_test_accuracy(self, Agent, file_name, logs_file_name):
        top_param = Agent.top_agr_params[0]
        trainer = Trainer(top_param, self.c_config, self.c_i, self.data_manager)
        accuracy = trainer.run()
        print("top_param = ", top_param)
        print("accuracy = ", Agent.top_rewards[0])
        with open(file_name, 'a') as f:
            f.write("top_param = " + str(top_param) + "\n accuracy : " + str(Agent.top_rewards[0]) + ", test_accuracy : " + str(accuracy))
        with open(logs_file_name, 'a') as f:
            f.write("top_param = " + str(top_param) + "\n accuracy : " + str(Agent.top_rewards[0]) + ", test_accuracy : " + str(accuracy))
        return accuracy

    def save_data_dict(self, filename_p=None):
        if filename_p:
            self.data_dict_file = filename_p
        data = pd.DataFrame(data=self.hot_method)
        data.to_csv(self.data_dict_file, index=False)
        data_length = len(self.hot_method["paras"])
        print("successfull !！！ save total ", data_length, " data!")
        return data_length

    def restore_data_dict(self):
        data_dict = pd.read_csv(self.data_dict_file, index_col=False)
        self.hot_method["paras"] = list(data_dict["paras"].values)
        self.hot_method["rewards"] = list(data_dict["rewards"].values)
        data_length = len(self.hot_method["paras"])
        print("successfull !！！ restore total ", data_length, " data!")
        return data_length


# 依据配置环境“ClassificationAlgorithmConfigureFile”，构建训练环境集
class EnvironmentManager(object):
    def __init__(self, data_manager):
        self.c_config = ClassificationAlgorithmConfigure()  # 分类算法配置信息
        self.data_manager = data_manager
        self.envs = []
        self.circle_counter = 0

    # 自动创建多个“单线程运行环境”
    def auto_create_multi_singleprocess_envs(self):
        """
        自动构建环境中的所有算法的训练方式
        :return: 环境中所有算法的个数,目前只有随机森林和XGBOOST，该值为2
        """
        for c_i in self.c_config.search_space:  # search_space=[1, 0]
            self.envs.append(SingleProcessEnvironment(self.data_manager, self.c_config, c_i))
        print("方法名：auto_create__multi_singleprocess_envs, 创建", len(self.envs), "个单进程训练环境！")
        return len(self.envs)

    def next_environment(self):
        """
        将Manager推向下一个算法的构建
        :return: env.state_dimensionality_list为形如[ 5 10  8  4  3]的列表，传递需要形成的lstm结构
                 env是下一个算法的接收器
                 env可用的方法有：run(actions_list), reset()
        """
        assert len(self.envs) > 0,  "环境列表为空，暂时还未创建任何环境！"
        env = self.envs[self.circle_counter]
        self.circle_counter += 1
        if self.circle_counter >= len(self.envs):
            self.circle_counter = 0
            print("EnvironmentManager.next_environment():所构建的训练环境已全部训练完成！")
        return env, env.state_dimensionality_list


if __name__ == "__main__":
    pass
