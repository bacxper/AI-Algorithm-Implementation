"""
简单实现BIRCH聚类算法
BIRCH（Balanced Iterative Reducing and Clustering using Hierarchies）
是一种用于大数据集的层次聚类算法。
它通过构建一个CF树（Clustering Feature Tree）来压缩数据集，然后可以对CF树进行进一步的聚类分析。
思路：
用面对对象的思想，先找到一类类对象，
CFNode和CFTree。
CF节点的变量有子节点数（n_points），孩子节点（children），线性和（linear_sum），平方和（square_sum）
CF节点的方法有更新节点情况（子节点树，线性和，平方和），计算两个节点之间的距离

CF树的变量有CF树的根节点（root），每个非叶节点的子节点数量限制分支因子（branchingfactor），插入新节点时考虑的阈值（threshold）
CF树的方法有插入新的节点（insert），
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 避免冲突
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 失败，不会