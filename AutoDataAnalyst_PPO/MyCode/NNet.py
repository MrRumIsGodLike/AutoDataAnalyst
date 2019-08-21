import tensorflow as tf
import numpy as np
import random
from datetime import datetime


# 构建预测网络
class NNet():
    # 定义预测网络图结构
    def __init__(self, input_param):
        self.INPUT_NODE = input_param  # 输入节点数
        self.OUTPUT_NODE = 1  # 输出节点数
        self.LAYER1_NODE = 16
        self.LAYER2_NODE = 16
        self.LAYER3_NODE = 8
        self.LAYER4_NODE = 16
        self.MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减系数
        self.REGULARIZATION_RATE = 0.0001  # 正则化项的权重系数
        self.LEARNING_RETE_BASE = 0.007  # 基学习率
        self.LEARNING_RETE_DECAY = 0.99  # 学习率的衰减率
        self.TRAIN_EPISODE = 100  # 训练的次数
        self.memory_size = 800  # 经验回放池大小
        self.batch_size = 8
        self.memory = []
        self.sess = tf.Session()
        self.path = "Mylog/test-meta4-no/10steps_90sample/run1"
        self.summary_writter = tf.summary.FileWriter(self.path, self.sess.graph)

        # 神经网络模型的训练过程
        # 输入值与标签值
        self.x = tf.placeholder(tf.float32, [None, self.INPUT_NODE], name='x-input')
        self.y_ = tf.placeholder(tf.float32, [None, self.OUTPUT_NODE], name='y-input')

        # 定义神经网络结构的参数
        # 第一层
        self.weights1 = tf.Variable(tf.truncated_normal([self.INPUT_NODE, self.LAYER1_NODE],
                                                        stddev=0.2))
        self.biases1 = tf.Variable(tf.constant(0.1, shape=[self.LAYER1_NODE]))

        # 第二层
        weights3 = tf.Variable(tf.truncated_normal([self.LAYER1_NODE, self.LAYER2_NODE],
                                                   stddev=0.2))
        biases3 = tf.Variable(tf.constant(0.1, shape=[self.LAYER2_NODE]))

        # 中间加一层看效果
        # weights4 = tf.Variable(tf.truncated_normal([self.LAYER2_NODE, self.LAYER3_NODE],
        #                                            stddev=0.2))
        # biases4 = tf.Variable(tf.constant(0.1, shape=[self.LAYER3_NODE]))

        # 输出层
        self.weights2 = tf.Variable(tf.truncated_normal([self.LAYER2_NODE, self.OUTPUT_NODE],
                                                        stddev=0.2))
        self.biases2 = tf.Variable(tf.constant(0.1, shape=[self.OUTPUT_NODE]))

        # 计算非滑动平均模型下的参数的前向传播的结果
        # y = self.inference(x, None, weights1, biases1, weights2, biases2)

        self.global_steps = tf.Variable(0, trainable=False)  # 定义存储当前迭代训练轮数的变量
        # 定义ExponentialMovingAverage类对象
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     self.MOVING_AVERAGE_DECAY, self.global_steps)  # 传入当前迭代轮数参数
        # 定义对所有可训练变量trainable_variables进行更新滑动平均值的操作op
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # 判断是否传入ExponentialMovingAverage类对象
        # if variable_averages == None:

        layer1 = tf.nn.relu(tf.matmul(self.x, self.weights1) + self.biases1)
        layer2 = tf.nn.relu(tf.matmul(layer1, weights3) + biases3)
        # layer3 = tf.nn.relu(tf.matmul(layer2, weights4) + biases4)
        self.y = tf.nn.sigmoid(tf.matmul(layer2, self.weights2) + self.biases2)

        # else:
        #     layer1 = tf.nn.sigmoid(tf.matmul(self.x, variable_averages.average(self.weights1))
        #                         + variable_averages.average(self.biases1))
        #     # layer2 = tf.nn.relu(tf.matmul(layer1, variable_averages.average(weights2))
        #     #                     + variable_averages.average(biases2))
        #     # layer3=  tf.nn.relu(tf.matmul(layer2, variable_averages.average(weights3))
        #     #                     + variable_averages.average(biases3))
        #     self.y=tf.nn.sigmoid(tf.matmul(layer1, variable_averages.average(self.weights2)) \
        #                          + variable_averages.average(self.biases2))

        # 计算滑动模型下的参数的前向传播的结果
        # self.y = self.inference(self.x, variable_averages, weights1, biases1, weights2, biases2)

        # 计算平方损失
        self.loss_mean = tf.reduce_mean(tf.square(self.y - self.y_))

        # 定义L2正则化器并对weights1和weights2正则化
        # regularizer = tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)
        # regularization = regularizer(self.weights1) + regularizer(self.weights2)
        # self.loss = loss_mean + regularization  # 总损失值

        # 定义指数衰减学习率
        self.learning_rate = tf.train.exponential_decay(
            self.LEARNING_RETE_BASE,
            self.global_steps,
            self.TRAIN_EPISODE,
            self.LEARNING_RETE_DECAY)

        # self.global_steps+=1

        # 定义梯度下降操作op，global_step参数可实现自加1运算
        self.train_step = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss_mean, global_step=self.global_steps)
        # self.train_step = tf.train.AdamOptimizer(self.learning_rate) \
        #     .minimize(self.loss,global_step=self.global_steps)
        # 组合两个操作op
        # self.train_op = tf.group(self.train_step, variables_averages_op)

        # 初始化网络训练
        self.sess.run(tf.global_variables_initializer())

    nnet_param_file = "../MyValidate_time/test-meta2-no/nnet_param.txt"

    def _init_net(self, sess, input_params):
        self.memory = []
        self.sess.run(tf.global_variables_initializer())
        biase2 = self.sess.run([self.biases2])
        print(biase2)
        # sess.run(tf.local_variables_initializer())
        a = sess

    # 建立经验回放池
    def store_transition(self, input_param, output_reward):
        # 存放一个batch_size(8个)的经验
        for i in range(self.batch_size):
            # 按水平方向进行堆叠
            transition = np.hstack((input_param[i], output_reward[i]))
            self.memory.append(transition)

    # 从经验池中随机选择数据进行训练,每次选择8个/1个batch_size
    def select_data(self):
        # 将一个列表的数据打乱
        random.shuffle(self.memory)
        batch_memory = self.memory[:self.batch_size]
        batch_memory = np.array(batch_memory).reshape(self.batch_size, self.INPUT_NODE + 1)
        input_param = batch_memory[:, :self.INPUT_NODE]
        output_reward = batch_memory[:, self.INPUT_NODE]
        return input_param, output_reward

    # 直接把训练池的数据全部训练
    def get_data(self):
        # 将一个列表的数据打乱
        random.shuffle(self.memory)
        memory = self.memory
        input = np.array(memory).reshape(len(memory), self.INPUT_NODE + 1)
        input_param = input[:, :self.INPUT_NODE]
        output_reward = input[:, self.INPUT_NODE]
        return input_param, output_reward

    # 输入数据归一化(线性)
    # def autoNorm(self, dataSet):
    #     minVals = dataSet.min(0)
    #     maxVals = dataSet.max(0)
    #     ranges = maxVals - minVals
    #     normDataSet = np.zeros(np.shape(dataSet))
    #     m = dataSet.shape[0]
    #     normDataSet = dataSet - np.tile(minVals, (m, 1))
    #     normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #     return normDataSet

    # 输入数据归一化(mean)
    # def autoMeanNorm(self, dataSet):
    #     for i in range(len(dataSet)):
    #         mean = np.mean(dataSet[i])
    #         std = np.std(dataSet[i])
    #         dataSet[i, :] = (dataSet[i, :] - mean) / std
    #     return dataSet

    # 从经验回放池中采样训练网络
    def train_net(self, sess, j):
        # 随机抽样8样本
        # input_x,output_y=self.select_data()
        # 直接全部训练
        input_x, output_y = self.get_data()
        # input_x = input_x.astype(np.float)
        output_y = np.array(output_y).reshape(len(output_y), 1)
        # for i in range(self.INPUT_NODE):
        #     mean=np.mean(input_x[:,i])
        #     max=np.max(input_x[:,i])
        #     min=np.min(input_x[:,i])
        #     input_x[:,i]=(input_x[:,i]-mean) / (np.std(input_x[:,i]))

        # 输入数据归一化(线性)
        # input_x = self.autoNorm(input_x.T).T

        # 输入数据归一化(Mean)
        # input_x = self.autoMeanNorm(input_x)

        # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

        # path = "Mylog/test-meta3-no/" + TIMESTAMP
        # path = "Mylog/test-meta3/" + TIMESTAMP

        loss_all = []
        # 训练j次/j个批次
        for i in range(j + 1):
            weight1 = self.sess.run([self.weights1])
            # y=sess.run([self.y])
            biase1 = self.sess.run([self.biases1])
            weight2 = self.sess.run([self.weights2])
            biase2 = self.sess.run([self.biases2])
            # rate=sess.run([self.learning_rate])
            # 喂入x,y数据
            input_feed = {self.x: input_x[i * 8:(i + 1) * 8, :], self.y_: output_y[i * 8:(i + 1) * 8, :]}
            _, loss1, y = self.sess.run([self.train_step, self.loss_mean, self.y], feed_dict=input_feed)

            # loss_all.append(loss1)

            # 将训练信息及结果写入nnet_param_file文件
            # with open(self.nnet_param_file, 'a') as f:
            #     f.write("\n train:" + str(j) + " step \n" +
            #             "x=" + str(input_x[i * 8:(i + 1) * 8, :]) + "\n" +
            #             "y_real= " + str(output_y[i * 8:(i + 1) * 8, :]) + "\n" + "y_train= " + str(y) + "\n" +
            #             "loss= " + str(loss1) + "\n" +
            #             ", weight1 = " + str(weight1) + "\n"
            #             + "biase1=" + str(biase1) + "\n"
            #             + "weight2 = " + str(weight2) + "\n" +
            #             "biase2 = " + str(biase2) + "\n")
        # 训练完毕一次记录一次实验结果
        # self.summarize(self.summary_writter, np.mean(loss_all), j, 'loss_meta')

    # 使用第三个数据集进行训练
    def pre_train_net(self, sess, j, input_x, output_y):
        output_y = np.array(output_y).reshape(len(output_y), 1)
        loss_all = []

        weight1 = self.sess.run([self.weights1])
        # y=sess.run([self.y])
        biase1 = self.sess.run([self.biases1])
        weight2 = self.sess.run([self.weights2])
        biase2 = self.sess.run([self.biases2])
        # rate=sess.run([self.learning_rate])
        # 喂入x,y数据
        input_feed = {self.x: input_x, self.y_: output_y}
        _, loss1, y = self.sess.run([self.train_step, self.loss_mean, self.y], feed_dict=input_feed)

        loss_all.append(loss1)

        # 将训练信息及结果写入nnet_param_file文件
        with open(self.nnet_param_file, 'a') as f:
            f.write("\n train:" + str(j) + " step \n" +
                    "x=" + str(input_x) + "\n" +
                    "y_real= " + str(output_y) + "\n" + "y_train= " + str(y) + "\n" +
                    "loss= " + str(loss1) + "\n" +
                    ", weight1 = " + str(weight1) + "\n"
                    + "biase1=" + str(biase1) + "\n"
                    + "weight2 = " + str(weight2) + "\n" +
                    "biase2 = " + str(biase2) + "\n")
        # 训练完毕一次记录一次实验结果
        self.summarize(sess, self.summary_writter, np.mean(loss_all), j, 'loss_train_meta')

    # 获取reward : y
    def get_reward(self, sess, input_x):
        # sess.run(tf.global_variables_initializer())
        # 喂入x数据
        input_feed = {self.x: input_x}
        reward = self.sess.run(self.y, feed_dict=input_feed)
        return reward

    def pre_get_reward(self, sess, j, input_x, output_y):
        input_feed = {self.x: input_x}
        reward = self.sess.run(self.y, feed_dict=input_feed)
        loss = np.mean(np.square(reward - output_y))
        self.summarize(sess, self.summary_writter, loss, j, 'loss_use_meta')
        return reward

    # 添加tensorboard记录信息
    def summarize(self, sess, summary_writter, value, step, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writter.add_summary(summary, step)
        summary_writter.flush()
