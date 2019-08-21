# coding: utf-8
# 文件种类：配置文件


# 系统信息配置类：用于配置该AutoDataAnalyst系统内，可供搜索的“分类算法”超参数信息；
class ClassificationAlgorithmConfigure(object):
    def __init__(self, search_space=[0]):
        self.search_space = search_space    # [0, 1]
        # 用于匹配算法与编号
        self.methods_dict = {0: ("RandomForestClassifier",
                                 ("n_estimators", (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)), # 12
                                 ("max_depth", (3, 6, 9, 12, 15, 18, 21, 24, 27, 30)), # 11
                                 ("min_samples_split", (2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)), # 21
                                 ("min_samples_leaf", (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)), # 21
                                 ("max_features", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)), # 9),  # 2
                                 ),
            1: ("XGBClassifier",
                ("max_depth", (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25)),  # 12
                ("learning_rate", (0.001, 0.005, 0.01, 0.04, 0.07, 0.1)),  # 6
                ("n_estimators", (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)),  # 13
                ("gamma", (0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)),  # 8
                ("min_child_weight", (1, 3, 5, 7)),  # 4
                ("subsample", (0.6, 0.7, 0.8, 0.9, 1.0)),  # 5
                ("colsample_bytree", (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),  # 6
                ("colsample_bylevel", (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),  # 6
                ("reg_alpha", (0, 0.1, 0.5, 1.0)),  # 4
                ("reg_lambda", (0.01, 0.03, 0.07, 0.1, 1.0))  # 5
                ),
            }

        # # 用于沟通policy-model和environment
        self.step_max_num = 0   # 5 + 8 = 13
        self.config_list = []   # [[5, 10, 8, 4, 3], [8, 7, 4, 5, 5, 4, 5, 5]]
        for x in self.search_space:  # search_space=[0, 1]
            key_value = self.methods_dict[x][1:]
            length = len(key_value)
            self.step_max_num += length
            method_small_list = [len(key_value[i][1]) for i in range(length)]
            self.config_list.append(method_small_list)
