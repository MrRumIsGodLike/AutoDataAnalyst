# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, space_eval
import tensorflow as tf
# from RL.AutoDataAnalyst_PPO.code.DataManager import DataManager
from MyCode.DataManager import DataManager
import time
import os

def t_main_3(data_manager, file_name, data_file_name, plot_data_path, data_dict_file):
    plot_data = {"time": [], "reward": []}

    global hot_method
    global data_cv, labels_cv
    global params, rewards
    global a
    a = 0
    hot_method = {"paras": [], "rewards": []}
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    data_manager = data_manager
    data_cv, labels_cv = data_manager.data_cv['data_cv'], data_manager.data_cv['labels_cv']
    data_test, labels_test = data_manager.data_cv['data_test'], data_manager.data_cv['labels_test']
    params = []
    rewards = []
    global times, start_time
    times = []
    summary_writter = tf.summary.FileWriter("log_tpe/", sess.graph)

    def save_data_dict(hot_method_p):
        data = pd.DataFrame(data=hot_method_p)
        data.to_csv(data_dict_file, index=False)
        data_length = len(hot_method_p["paras"])
        print("successfull !！！ save total ", data_length, " data!")
        return data_length

    def restore_data_dict():
        global hot_method
        data_dict = pd.read_csv(data_dict_file, index_col=False)
        hot_method["paras"] = list(data_dict["paras"].values)
        hot_method["rewards"] = list(data_dict["rewards"].values)
        data_length = len(hot_method["paras"])
        print("successfull !！！ restore total ", data_length, " data!")
        return data_length

    # 未通过scalar的，使用该函数
    def summarize(summary_writter, value, step, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writter.add_summary(summary, step)
        summary_writter.flush()

    def func(args, flag=None):
        global hot_method, start_time
        global params, times, rewards
        global data_cv, labels_cv
        global a
        n_estimators = args["n_estimators"]
        max_depth = args["max_depth"]
        min_samples_split = args["min_samples_split"]
        min_samples_leaf = args["min_samples_leaf"]
        max_features = args["max_features"]

        agr_params = [int(n_estimators), int(max_depth), int(min_samples_split)
            , int(min_samples_leaf), float(max_features)]
        agr_params = np.array(agr_params).reshape(1, len(agr_params))
        val = 0
        if np.any(str(agr_params[0]) in hot_method["paras"]):
            index = hot_method["paras"].index(str(agr_params[0]))
            val = hot_method["rewards"][index]
            print("if!!!")
        else:
            print("else!!!")
            rfc = RandomForestClassifier(n_estimators=int(n_estimators),
                                         max_depth=int(max_depth),
                                         min_samples_split=int(min_samples_split),
                                         min_samples_leaf=int(min_samples_leaf),
                                         max_features=float(max_features),
                                         bootstrap=True,
                                         n_jobs=-1)
            results = cross_val_score(rfc, data_cv, labels_cv, cv=2, n_jobs=1)
            val = np.mean(results)
            hot_method["paras"].append(str(agr_params[0]))
            hot_method["rewards"].append(val)
        print("val:" + str(val))
        if flag == None:
            params.append(args)
            time_p = time.time()
            times.append(time_p - start_time)
            rewards.append(val)
            # times.append(time_p - start_time)

            plot_data["time"].append(time_p - start_time)
            plot_data["reward"].append(val)
            a = a + 1
            if a % 50 == 0:
                data = pd.DataFrame(plot_data)
                data.to_csv(plot_data_path, index=False)
            summarize(summary_writter, val, a, 'max_reward')
        return -val

    space = {
        'n_estimators': hp.uniform('n_estimators', 10, 1000),
        'max_depth': hp.uniform('max_depth', 1, 35),
        'min_samples_split': hp.uniform('min_samples_split', 2, 100),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 1, 100),
        'max_features': hp.uniform('max_features', 0.1, 0.9)  # 2
    }
    # ------------huifu lishi shuju---------------;
    # restore_data_dict()
    start_time = time.time()
    with open(file_name, 'a') as f:
        f.write("start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
    best = fmin(func, space, algo=tpe.suggest, max_evals=1)
    params = pd.DataFrame(params)
    params["time"] = times
    params["accuracy"] = rewards
    params.to_csv(data_file_name, index=False)

    def print_test_accuracy(args, data_cv, labels_cv, data_test, labels_test):
        rfc = RandomForestClassifier(n_estimators=int(args['n_estimators']),
                                     max_depth=int(args['max_depth']),
                                     min_samples_split=int(args['min_samples_split']),
                                     min_samples_leaf=int(args['min_samples_leaf']),
                                     max_features=float(args['max_features']),
                                     bootstrap=True,
                                     n_jobs=-1)
        rfc.fit(data_cv, labels_cv)
        val = rfc.score(data_test, labels_test)
        return val

    test_accuracy = print_test_accuracy(space_eval(space, best), data_cv, labels_cv, data_test, labels_test)

    with open(file_name, 'a') as f:
        f.write("\n params=\n " + str(params))
    with open(file_name, 'a') as f:
        f.write("\n best_action_index= " + str(best) + "\n best_action_param= " + str(
            space_eval(space, best)) + "\n best_action_accuracy= " + str(
            func(space_eval(space, best), 1)) + "\n test_accuracy= " + str(test_accuracy))
    over_time = time.time()
    sum_time = over_time - start_time
    with open(file_name, 'a') as f:
        f.write("\n finish ---- search hyperParams of the algorithm ," + "start_time= " + str(start_time) +
                ", over_time= " + str(over_time) + ", sum_time = " + str(sum_time) + "\n")
    # -------------------baocun lishi shuju-----------------------;
    save_data_dict(hot_method)

    print("best_action_index", best)
    print("-----best_action_param:", space_eval(space, best))
    print("-----best_action_accuracy,", func(space_eval(space, best), 1))
    print('RFC, test_accuracy=', test_accuracy)
    print("----------TPE 算法运行结束！----------")
    # print("params:", params)
    del data_cv, labels_cv, data_test, labels_test
    del params, rewards, times, start_time


# if __name__ == '__main__':
#     data_manager = DataManager()
#     os.mkdir("../validate_time/params_data_tpe/digits_tpe_" + str(1))
#     os.mkdir("./data_dict/digits_tpe_" + str(1))
#     path = "digits_tpe_" + str(1)
#     file_name = "../validate_time/params_data_tpe/" + path + "/logs.txt"
#     data_file_name = "../validate_time/params_data_tpe/" + path + "/data.txt"
#     plot_data_path = "../validate_time/params_data_tpe/" + path + "/plot_data.csv"
#     data_dict_file = "./data_dict/" + path + "/t_tpe.csv"
#     t_main_3(data_manager, file_name, data_file_name, plot_data_path, data_dict_file)
