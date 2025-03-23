"""
K-均值聚类算法
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 避免冲突
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 准备数据
data = [
    [-0.52, 1.8539],
    [2.5849, 2.2481],
    [0.9919, 1.9234],
    [2.9443, 3.7382],
    [-0.424, 3.622],
    [1.7762, 2.6264],
    [2.0581, 2.0918],
    [1.5754, 1.1924],
    [1.7971, 1.5387],
    [0.4869, 0.594],
    [7.8736, 7.6255],
    [8.185, 7.5291],
    [9.3666, 9.7513],
    [8.4139, 8.7532],
    [10.5374, 8.065],
    [9.1401, 7.7072],
    [7.1372, 8.0828],
    [8.5458, 8.7662],
    [8.3479, 10.2368],
    [9.1033, 8.3269],
    [3.7794, 4.8633],
    [3.721, 4.6794],
    [3.2663, 4.5548],
    [3.9355, 5.0016],
    [2.5560, 5.2594],
    [4.6123, 4.0442],
    [2.6765, 3.6859],
    [3.3384, 4.2267]
]


# 定义距离函数
def dfunc(x_1, x_2):
    return ((x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2) ** (1 / 2)


# 计算聚类中心函数
def getCenter(k, Clu, ):
    mii = []
    for i in range(k):
        sum_x = sum(point[0] for point in Clu[i])
        sum_y = sum(point[1] for point in Clu[i])
        num_points = len(Clu[i])
        center_x = sum_x / num_points
        center_y = sum_y / num_points
        mii.append([center_x, center_y])
    return mii


flag = 1

# 聚类簇数目与聚类
k = 3  # 观察图主观得到
Clustering = []
for i in range(k):
    Clustering.append([])
Clustering2 = []

# 选择k个初始中心
mi = [
    [1.5, 2],
    [4, 5],
    [9, 9]
]  # 观察图主观选取得到

# 分配样本点
for i in range(len(data)):
    temp = []
    for j in range(k):
        temp.append(dfunc(mi[j], data[i]))
    l = temp.index(min(temp))
    Clustering[l].append(data[i])

while 1:
    # 计算新的聚类中心
    if flag % 2 == 1:
        mi = getCenter(k, Clustering)

        # 重新分配样本点
        Clustering2 = []
        for i in range(k):
            Clustering2.append([])
        for i in range(len(data)):
            temp = []
            for j in range(k):
                temp.append(dfunc(mi[j], data[i]))
            Clustering2[temp.index(min(temp))].append(data[i])
    else:
        mi = getCenter(k, Clustering2)

        # 重新分配样本点
        Clustering = []
        for i in range(k):
            Clustering.append([])
        for i in range(len(data)):
            temp = []
            for j in range(k):
                temp.append(dfunc(mi[j], data[i]))
            Clustering[temp.index(min(temp))].append(data[i])

    # 检查是否停止循环
    if Clustering == Clustering2:
        break
    else:
        flag += 1
        continue

for i in range(k):
    print(Clustering[i])

# 提取x和y
x = [d[0] for d in data]
y = [d[1] for d in data]
x1, y1 = zip(*Clustering[0])
x2, y2 = zip(*Clustering[1])
x3, y3 = zip(*Clustering[2])

# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='red')
# plt.title('原图')
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.grid(True)
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 第一张图
ax1.scatter(x, y, color='red', label='未聚类')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('原图')

# 第二张图
ax2.scatter(x1, y1, color='blue', label='第一类')
ax2.scatter(x2, y2, color='purple', label='第二类')
ax2.scatter(x3, y3, color='green', label='第三类')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('聚类后')

# 添加图例，loc指定图例的一个角落，把这个角落放在bbox_to_anchor指定的位置，(1,1)就是图的右上角了
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 让图更紧凑
plt.tight_layout()

plt.show()
