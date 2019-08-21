# coding:utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
# from xgboost.sklearn import
from bayes_opt import BayesianOptimization
# from RL.AutoDataAnalyst_PPO.code.DataManager import DataManager
from MyCode.DataManager import DataManager

# import timeXGBClassifier
import os

def t_main_2(data_manager,file_name,data_file_name,plot_data_path,data_dict_file):
    plot_data={"time":[],"reward":[]}
    global hot_method
    global data_cv, labels_cv, data_test, labels_test
    global a
    a=0
    hot_method = {"paras": [], "rewards": []}
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    data_manager = data_manager
    data_cv, labels_cv = data_manager.data_cv['data_cv'], data_manager.data_cv['labels_cv']
    data_test, labels_test = data_manager.data_cv['data_test'], data_manager.data_cv['labels_test']
    global times, start_time
    times = []
    summary_writter = tf.summary.FileWriter("log_gp/", sess.graph)
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

    # 随机森林
    def rfc(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        global hot_method
        global a
        agr_params = [int(n_estimators), int(max_depth), int(min_samples_split)
            , int(min_samples_leaf), float(max_features)]
        agr_params = np.array(agr_params).reshape(1, len(agr_params))
        val = 0
        if np.any(str(agr_params[0]) in hot_method["paras"]):
            index = hot_method["paras"].index(str(agr_params[0]))
            val = hot_method["rewards"][index]
        else:
            val = cross_val_score(RFC(n_estimators=int(n_estimators),
                                       max_depth=int(max_depth),
                                       min_samples_split=int(min_samples_split),
                                       min_samples_leaf=int(min_samples_leaf),
                                       max_features=float(max_features),
                                       bootstrap=True,
                                       n_jobs=-1),
                                   data_cv, labels_cv, cv=2, n_jobs=1).mean()
            hot_method["paras"].append(str(agr_params[0]))
            hot_method["rewards"].append(val)
        global times, start_time
        time_p = time.time()
        times.append(time_p-start_time)

        plot_data["time"].append(time_p-start_time)
        plot_data["reward"].append(val)
        a=a+1
        if a%50==0:
            data=pd.DataFrame(plot_data)
            data.to_csv(plot_data_path, index=False)

        summarize(summary_writter, val, a, 'reward')
        return val

    gp_params = {"alpha": 1e-5}
    # ------------huifu lishi shuju---------------;
    # restore_data_dict()
    start_time = time.time()
    with open(file_name, 'a') as f:
        f.write("start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
    rfcBO = BayesianOptimization(rfc, {'n_estimators': (10, 1000), 'max_depth': (1, 35), 'min_samples_split': (2, 100),
                                       'min_samples_leaf': (1, 100), 'max_features': (0.1, 0.9)},
                                 verbose=1)
    rfcBO.maximize(n_iter=1, **gp_params)
    params = pd.DataFrame(rfcBO.res['all']['params'])
    params["time"] = times[5:]
    params["accuracy"] = rfcBO.res['all']['values']
    params.to_csv(data_file_name, index=False)



    def print_test_accuracy(args, data_cv, labels_cv, data_test, labels_test):
        rfc = RFC(n_estimators=int(args['n_estimators']),
                  max_depth=int(args['max_depth']),
                  min_samples_split=int(args['min_samples_split']),
                  min_samples_leaf=int(args['min_samples_leaf']),
                  max_features=float(args['max_features']),
                  bootstrap=True,
                  n_jobs=-1)
        rfc.fit(data_cv, labels_cv)
        val = rfc.score(data_test, labels_test)
        return val

    test_accuracy = print_test_accuracy(rfcBO.res['max']['max_params'], data_cv, labels_cv, data_test, labels_test)
    with open(file_name, 'a') as f:
        f.write("\n params=\n " + str(params))
    with open(file_name, 'a') as f:
        f.write("\n best_action_param= " + str(rfcBO.res['max']['max_params']) + "\n best_action_accuracy= " + str(
            rfcBO.res['max']['max_val']) + "\n test_accuracy= " + str(test_accuracy))
    over_time = time.time()
    sum_time = over_time - start_time
    with open(file_name, 'a') as f:
        f.write("\n finish ---- search hyperParams of the algorithm ," + "start_time= " + str(start_time) +
                ", over_time= " + str(over_time) + ", sum_time = " + str(sum_time) + "\n")
    # -------------------baocun lishi shuju-----------------------;
    save_data_dict(hot_method)

    print('Final Results')
    print('RFC:', rfcBO.res['max']['max_val'])
    print('RFC:', rfcBO.res['max']['max_params'])
    # print('RFC:', rfcBO.res['all'])
    print('RFC, test_accuracy=', test_accuracy)
    print("----------GP 算法运行结束！----------")
    del data_cv, labels_cv, data_test, labels_test
    del params

# if __name__ == "__main__":
    # data_manager = DataManager()
    # os.mkdir("../validate_time/params_data_gp/digits_gp_" + str(2))
    # os.mkdir("./data_dict/digits_gp_" + str(2))
    # path = "digits_gp_" + str(2)
    # file_name = "../validate_time/params_data_gp/" + path + "/logs.txt"
    # data_file_name = "../validate_time/params_data_gp/" + path + "/data.txt"
    # plot_data_path = "../validate_time/params_data_gp/" + path + "/plot_data.csv"
    # data_dict_file = "./data_dict/" + path + "/t_gp.csv"
    # t_main_2(data_manager, file_name, data_file_name, plot_data_path, data_dict_file)
