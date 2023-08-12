# 心跳信号分类预测

## [样本数据集](https://tianchi.aliyun.com/competition/entrance/531883/information)

|       Field       |      **Description**       |
| :---------------: | :------------------------: |
|        id         |  为心跳信号分配的唯一标识  |
| heartbeat_signals |        心跳信号序列        |
|       label       | 心跳信号类别（0、1、2、3） |

## Dependencies:

- Python 3.4 or greater
- tensorflow2.2+

## 文件/文件夹及其用途：

- cnn.ipnb：卷积神经网络
- data_analysis.ipynb：数据特征分析
- lightgbm.ipynb：lightgbm实现分类