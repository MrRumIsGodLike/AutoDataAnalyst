# coding: utf-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier

# from RL.AutoDataAnalyst_PPO.code.DataManager import DataManager
# from RL.AutoDataAnalyst_PPO.code.configFile.ClassificationAlgorithmConfigureFile import ClassificationAlgorithmConfigure
from MyCode.DataManager import DataManager
from MyCode.configFile.ClassificationAlgorithmConfigureFile import ClassificationAlgorithmConfigure

class Trainer(object):
    def __init__(self, actions, c_config, c_i, data_manager):
        self.actions = actions
        self.c_config = c_config
        self.c_i = c_i
        self.data_manager = data_manager

        self.method = None
        # param = self.getparams()
        if c_i == 0:
            self.method = RandomForestClassifier(
                n_estimators=int(actions[0]),
                max_depth=int(actions[1]),
                min_samples_split=int(actions[2]),
                min_samples_leaf=int(actions[3]),
                max_features=actions[4],
                bootstrap=True,
                n_jobs=-1)
        elif c_i == 1:
            self.method = XGBClassifier(
                max_depth=int(actions[0]),
                learning_rate=float(actions[1]),
                n_estimators=int(actions[2]),
                gamma=float(actions[3]),
                min_child_weight=int(actions[4]),
                subsample=float(actions[5]),
                colsample_bytree=float(actions[6]),
                colsample_bylevel=float(actions[7]),
                reg_alpha=float(actions[8]),
                reg_lambda=float(actions[9]),
                nthread=-1)
        else:
            assert False, "Trainer.__init__: 异常信息！"

    def getparams(self):
        param = []
        key_value = self.c_config.methods_dict[self.c_i][1:]
        assert len(self.actions) == len(key_value), "Trainer.getparams: 数据维度应该一样！"
        for i in range(len(key_value)):
            param.append(key_value[i][1][self.actions[i]])
        return param

    def run(self):
        self.fit()
        accuracy = self.estimate()
        return accuracy

    # 交叉验证集版本的训练方法
    def run_CV(self):
        results = cross_val_score(self.method, self.data_manager.data_cv['data_cv'], self.data_manager.data_cv['labels_cv'], cv=2, n_jobs=1)
        accuracy = np.mean(results)
        return accuracy

    def fit(self):
        self.method.fit(self.data_manager.data_cv['data_cv'], self.data_manager.data_cv['labels_cv'])

    def predict(self, x):
        return self.method.predict(x)

    def estimate(self):
        return self.method.score(self.data_manager.data_cv["data_test"], self.data_manager.data_cv["labels_test"])


def test(data):
    # actions = [0, 3, 3, 0, 0]
    # 375.2.16.11.0.34941125
    # [1200.     35.      2.      1.      0.1]
    actions = [1100, 27, 2, 1, 0.2]
    c_config = ClassificationAlgorithmConfigure()
    print("test-----1")
    c_i = 0
    #data_manager = DataManager()
    print("test-----2")
    trainer1 = Trainer(actions, c_config, c_i, data)
    print("test-----3")
    # accuracy1 = trainer1.run()
    accuracy1 = trainer1.run_CV()
    print("RandomForestClassifier, accuracy1=", accuracy1)

    # actions = [1, 2, 3, 0, 1, 0, 3, 3]
    # c_i = 1
    # print("test-----4")
    # trainer2 = Trainer(actions, c_config, c_i, data_manager)
    # print("test-----5")
    # # accuracy2 = trainer2.run()
    # accuracy2 = trainer2.run_CV()
    # print("XGBClassifier, accuracy2=", accuracy2)

if __name__ == '__main__':
    data_manager = DataManager()
    test(data_manager)