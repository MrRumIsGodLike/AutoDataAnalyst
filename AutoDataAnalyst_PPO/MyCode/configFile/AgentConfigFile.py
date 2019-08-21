# coding: utf-8


# 类的功能：LSTM的超参数设置
class AgentConfig:
    n_layers = 3                  # lstm层数
    n_hidden_units = 35           # 每层结构中,神经元的个数neurons in hidden layer
    # dr = 0.95
    dr = 0.8
    lr = 0.00007  # 学习率learning rate（ADAM Optimizer）
    lr_ = 0.01  #学习率learning rate (addsign optimizer)
    batch_size = 8
    pre_size = 5

