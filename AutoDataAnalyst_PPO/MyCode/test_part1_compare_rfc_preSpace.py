# -*- coding: utf-8 -*-
# from RL import DataManager
from MyCode.DataManager import DataManager

# ---------读取数据---------------
print("1. 读取数据! ---------------")
data_manager = DataManager()

# ---------------starting-validate-time---------------------
print("---starting-validate-time--测试第2个环境：GP 选择！")
t_main_2(data_manager)
print("第2个环境测试完成！")

print("---starting-validate-time--测试第1个环境：Agent(chen)选择！")
t_main_1(data_manager)
print("第1个环境测试完成！")

print("---starting-validate-time--测试第3个环境：TPE 选择！")
t_main_3(data_manager)
print("第3个环境测试完成！")

print("---starting-validate-time--测试第4个环境：Rand 选择！")
t_main_4(data_manager)
print("第4个环境测试完成！")
# ---------------finished-validate-time---------------------

# ---------------starting-validate-accuracy---------------------
print("---starting-validate-accuracy--测试第1个环境：Agent(chen)选择！")
a_main_1(data_manager)
print("第1个环境测试完成！")

print("---starting-validate-accuracy--测试第2个环境：GP 选择！")
a_main_2(data_manager)
print("第2个环境测试完成！")

print("---starting-validate-accuracy--测试第3个环境：TPE 选择！")
a_main_3(data_manager)
print("第3个环境测试完成！")

print("---starting-validate-accuracy--测试第4个环境：Rand 选择！")
a_main_4(data_manager)
print("第4个环境测试完成！")
# ---------------finished-validate-accuracy---------------------







