import numpy as np

#
class RandomForestCliper:
    def __init__(self):
        self.n_estimators = [100, 1200]
        self.max_depth = [1, 35]
        self.min_samples_split = [2, 100]
        self.min_samples_leaf = [1, 100]
        self.max_features = [0.1, 0.9]

    def convert_to_action(self, sample_action, i, dist_param):
        action = []
        if i == 0:
            action = [int(np.clip(
                (self.n_estimators[1] - self.n_estimators[0]) / 2 * s[0][0] + ((self.n_estimators[1] - self.n_estimators[0]) / 2 + self.n_estimators[0]),
                100, 1200)) for s in sample_action]
        elif i == 1:
            action = [int(np.clip(
                (self.max_depth[1] - self.max_depth[0]) / 2 * s[0][0] + ((self.max_depth[1] - self.max_depth[0]) / 2 + self.max_depth[0]),
                1, 35))
                for s in sample_action]
        elif i == 2:
            action = [int(np.clip(
                (self.min_samples_split[1] - self.min_samples_split[0]) / 2 * s[0][0]  + ((self.min_samples_split[1] - self.min_samples_split[0]) / 2+self.min_samples_split[0]),
                2, 100))
                for s in sample_action]
        elif i == 3:
            action = [int(np.clip(
                (self.min_samples_leaf[1] - self.min_samples_leaf[0]) / 2 * s[0][0] + ((self.min_samples_leaf[1] - self.min_samples_leaf[0]) / 2+self.min_samples_leaf[0]),
                1, 100))
                for s in sample_action]
        elif i == 4:
            action = [float(np.clip(
                (self.max_features[1] - self.max_features[0]) / 2*s[0][0] + ((self.max_features[1] - self.max_features[0]) / 2+self.max_features[0]),
                0.1, 0.9))
                for s in sample_action]
        return action

    # def convert_to_action(self,param,i):
    #     if i == 0:
    #         action = [int(np.clip(i,100,1200)) for i in param]
    #     elif i == 1:
    #         action = [int(np.clip(i,1,35)) for i in param]
    #     elif i == 2:
    #         action = [int(np.clip(i,2,100)) for i in param]
    #     elif i == 3:
    #         action = [int(np.clip(i,1,100)) for i in param]
    #     elif i == 4:
    #         action = [float(np.clip(i,0.1,0.9)) for i in param]
    #     return action

    # def convert_to_action(self,param,i):
    #     if i == 0:
    #         action = [int(i) for i in param*(self.n_estimators[1]-self.n_estimators[0])+self.n_estimators[0]]
    #     elif i == 1:
    #         action = [int(i) for i in param * (self.max_depth[1] - self.max_depth[0]) + self.max_depth[0]]
    #     elif i == 2:
    #         action = [int(i) for i in param * (self.min_samples_split[1] - self.min_samples_split[0]) + self.min_samples_split[0]]
    #     elif i == 3:
    #         action = [int(i) for i in param * (self.min_samples_leaf[1] - self.min_samples_leaf[0]) + self.min_samples_leaf[0]]
    #     elif i == 4:
    #         action = [float(i) for i in param * (self.max_features[1] - self.max_features[0]) + self.max_features[0]]
    #     return action
