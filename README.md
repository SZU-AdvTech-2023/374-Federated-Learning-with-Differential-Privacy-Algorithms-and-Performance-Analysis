# **1. 实验环境**

## 1.1 Python环境
本实验代码使用python语言编写，python版本为3.8.16，并基于pytorch进行模型训练，pytorch版本为1.13.1。

## 1.2 Python包

本实验代码需安装numpy和math包用于数学计算，安装matplotlib包用于绘图。
# **2. 文件介绍**

* model.py: 存储MNIST数据集的训练模型MLP；
* Client.py: 存储客户端的模型训练过程；
* Picture\_1\_diff\_epsilon\_all\_client.py，  
Picture\_2\_diff\_epsilon\_all\_client\_small.py，  
Picture\_3\_diff\_epsilon\_partial\_client.py，  
Picture\_4\_diff\_C\_all\_client.py，  
Picture\_5\_diff\_N\_all\_client.py，  
Picture\_6\_diff\_T\_all\_client.py，  
Picture\_7\_diff\_K\_partial\_client.py：  
上述7个文件分别用于运行课程论文中的图2-8的实验设置，可以直接run运行；
* Plot\_Picture\_1\_diff\_epsilon\_all\_client.py，  
Plot\_Picture\_2\_diff\_epsilon\_all\_client\_small.py，  
Plot\_Picture\_3\_diff\_epsilon\_partial\_client.py，   
Plot\_Picture\_4\_diff\_C\_all\_client.py，  
Plot\_Picture\_5\_diff\_N\_all\_client.py，  
Plot\_Picture\_6\_diff\_T\_all\_client.py，  
Plot\_Picture\_7\_diff\_K\_partial\_client.py：  
上述7个文件分别用于生成课程论文中的图2-8，可以直接run运行。