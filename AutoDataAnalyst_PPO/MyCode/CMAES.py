# coding: utf-8
import numpy as np
import os
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# from RL.AutoDataAnalyst_PPO.code.DataManager import DataManager
from MyCode.DataManager import DataManager
import chocolate as choco
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from chocolate import SQLiteConnection


def CMAES(data_manager,n,file_name,conn):
    data_cv, labels_cv = data_manager.data_cv['data_cv'], data_manager.data_cv['labels_cv']
    data_test, labels_test = data_manager.data_cv['data_test'], data_manager.data_cv['labels_test']

    def score_gbt(params):
        rfc = RandomForestClassifier(**params)
        results = cross_val_score(rfc, data_cv, labels_cv, cv=2, n_jobs=1)
        val = np.mean(results)
        return -val


    space = {
        'n_estimators': choco.uniform(10,1000), # 12
        'max_depth': choco.uniform(1,35),  # 11
        'min_samples_split': choco.uniform(2,100),  # 21
        'min_samples_leaf': choco.uniform(1,100),  # 21
        'max_features': choco.uniform(0.1,0.9)
    }
    sampler = choco.CMAES(conn, space)
    plot_data = {"time": [], "reward": [], "param": []}
    start_time = time.time()
    for i in range(n):
        token, params = sampler.next()
        print(params)
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        loss = score_gbt(params)
        print(loss)
        sampler.update(token, loss)
        step_time=time.time()
        ont_time=step_time-start_time
        plot_data["time"].append(ont_time)
        plot_data["reward"].append(-loss)
        plot_data["param"].append(params)
        plot = pd.DataFrame(data=plot_data)
        plot.to_csv(file_name, index=False)
data_manager = DataManager(4)
if __name__=='__main__':
    n=1
    # conn = choco.SQLiteConnection(url="sqlite:///db.db")
    # results = conn.results_as_dataframe()
    # results = results['_loss']
    # results=np.array(results).reshape(len(results),1)
    # plot_time_reward = "../data.csv"
    # plot = pd.DataFrame(data=results)
    # plot.to_csv(plot_time_reward, index=False)
    # for i in range(0,2):
    #     j=0
    os.mkdir("../validate_time/params_data_cmaes/" + str(0) + "_cmaes_" + str(0))
    path = str(0) + "_cmaes_" + str(0)
    file_name = "../validate_time/params_data_cmaes/" + path + "/data.csv"
    url_path = "sqlite:///"+"../validate_time/params_data_cmaes/" + path +"/mnistdb.db"
    conn = choco.SQLiteConnection(url=url_path)
    CMAES(data_manager, n, file_name, conn)
        # os.mkdir("./val/mnist_cmaes_"+str(i))
        # path="mnist_cmaes_"+str(i)
        # file_name = "./val/"+path+"/data.csv"
        # url_path="sqlite:///mnistdb.db"+str(i)
        # conn = choco.SQLiteConnection(url=url_path)
        # CMAES(data_manager,n,file_name,conn)