"""
利用K-均值算法，对输入的彩色图片进行处理。
"""

# 用于创建GUI
import tkinter as tk

# 用于与用户交互以选择文件或目录的对话框。
from tkinter import filedialog

import numpy as np

import matplotlib

# 指定matplotlib使用TkAgg作为后端，这样matplotlib的图形就可以嵌入到Tkinter窗口中。
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt

# Python Imaging Library，用于处理图像。
from PIL import Image, ImageTk

import cv2


def k_means_clustering(image, K, mode):
    # 假设 image 是您读取的图像
    # .shape是NumPy库的，用于返回数组的维度信息，具体来说返回一个元组，其中包含数组各个维度的长度
    # 对于一个二维数组，.shape会返回(高度,宽度)。对于三维数组，.shape会返回(高度,宽度,颜色通道数)
    if len(image.shape) == 2:
        # 图像是灰度的，需要转换为 BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] != 3:
        # 图像不是三通道的，需要处理
        print("输入图像通道数不正确:", image.shape)

    if mode == "mode1":
        lab_image = image
    else:
        # 将图像从BGR颜色空间转换到LAB颜色空间。因为Lab颜色空间的欧氏距离更能反映人眼对颜色的感知（不懂）。
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    #  reshape图像为二维数组，其中每行代表一个像素点的颜色值
    # 举个例子
    # 原本是[[[1,2,3],[2,3,4]],[[2,4,1],[1,6,5]]]
    # reshape之后是[[1,2,3],[2,3,4],[2,4,1],[1,6,5]]
    # pixel：像素
    # -1是自动计算行数，以保持与原始图像中像素点的数量一致，3 表示每个像素点有三个颜色通道。
    pixel_values = lab_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 定义停止准则
    # criteria：准则，标准
    # cv2.TERM_CRITERIA_EPS 表示当两次迭代之间的误差小于某个阈值时停止。
    # cv2.TERM_CRITERIA_MAX_ITER 表示达到最大迭代次数时停止。
    # 这两个标志通过按位或操作 (+) 结合在一起，表示满足任一条件即可停止算法。
    # 参数 (100, 0.85) 分别表示：
    # 最大迭代次数为 100 次。
    # 误差阈值设置为 0.85。
    # criteria: 终止准则，由三个值组成的元组 (type, max_iter, epsilon):
    # type: 可以是 cv2.TERM_CRITERIA_EPS, cv2.TERM_CRITERIA_MAX_ITER, 或两者的组合 cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER。
    # max_iter: 最大迭代次数。
    # epsilon: 精度要求，当聚类中心的变化小于此值时，算法终止。
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # 应用KMeans算法
    # _：
    # 这个变量通常被忽略，它包含了每次迭代后聚类中心的日志。由于在这里没有被使用，所以用下划线_来表示。
    # labels：
    # 这是一个数组，包含了每个像素点被分配到的聚类中心的索引。例如，如果labels[i] == j，那么第i个像素点被分配到了第j个聚类中心。实际label应当是二维数组
    # centers：
    # 这是一个数组，包含了最终的聚类中心。每个聚类中心也是一个数组，其长度等于颜色通道数，表示该聚类中心在颜色空间中的位置。
    # bestLabels: 用于存储最佳结果的标签数组。如果为 None，则不使用此功能。
    # cv2.KMEANS_RANDOM_CENTERS：这是一个标志，告诉函数从数据中随机选择初始中心。
    # attempts: 尝试的次数。kmeans 算法可能会因为初始聚类中心的选择而导致局部最优解。通过多次尝试不同的初始中心，可以提高找到全局最优解的概率。
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将聚类标签转换回uint8类型
    centers = np.uint8(centers)
    # centers[labels.flatten()] 是一个索引操作，它根据labels数组中的索引从centers数组中选择对应的聚类中心。
    # 由于labels已经被展平为一维数组，这个操作将为每个像素点分配一个聚类中心的颜色值。
    # 总的描述：遍历labels，labels的每个值是第j个聚类中心的那个j，centers[j]就取到第j个聚类中心的颜色值了。
    segmented_data = centers[labels.flatten()]

    # 重新reshape成原始图像大小
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image


def show_images(image, segmented_image):
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Segmented Image')
    plt.show()


def open_image():
    # 打开文件选择对话框
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    # 打开并显示图像
    image = Image.open(filepath)
    image_array = np.array(image)

    K = int(k_entry.get())
    mode = selected_mode.get()
    # 应用直方图均衡化
    segmented_image = k_means_clustering(image_array, K, mode)
    # 显示图像
    show_images(image_array, segmented_image)


# 创建主窗口
root = tk.Tk()
root.title("K-Means Clustering on Image")

# tk.Label 是 Tkinter 中的一个组件，用于显示不可编辑的文本。root体现放在根窗口。
k_label = tk.Label(root, text="Enter the value of K:")
k_label.pack()

# tk.Entry 是 Tkinter 中的一个组件，用于创建一个单行文本输入框，允许用户输入文本。
k_entry = tk.Entry(root)
k_entry.pack()

# .insert() 方法用于在 Entry 组件中插入文本。
# 第一个参数 0 表示文本插入的位置，0 表示插入到文本框的最开始位置。
# "2" 是要插入的文本，这里是一个字符串 "2"，它作为输入框的默认值。
k_entry.insert(0, "2")  # 默认值

# 定义一个变量，用于存储用户的选择
selected_mode = tk.StringVar()

modes = [("BGR空间", "mode1"), ("LAB空间", "mode2")]
for text, mode in modes:
    rb = tk.Radiobutton(root, text=text, variable=selected_mode, value=mode)
    rb.pack(anchor='w')

# 创建一个按钮来打开文件选择对话框
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()  # 用于将组件添加到父窗口中

# 运行主循环
root.mainloop()
