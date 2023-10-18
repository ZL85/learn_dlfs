# 神经网络可以自动地从数据中学习到合适的权重参数
# 输入层
# 隐藏层
# 输出层


# 激活函数(activation function)
# a = b + w1 * x1 + w2 * x2  # 计算加权输入信号和偏置的总和，记为a
# y = h(a)  # 用h()函数将a转换为输出y


# 阶跃函数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


import numpy as np


def step_function(x):  # 修改成支持numpy数组的实现
    y = x > 0
    return y.astype(np.int)


x = np.array([-1.0, 1.0, 2.0])
print(x)
y = x > 0
print(y)
y = y.astype(int)
print(y)

# 画出阶跃函数
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # y轴标签
plt.title("step_function")  # 标题
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.xticks(np.arange(-6, 6.1, 1))  # 设置x轴刻度
plt.show()
# 阶跃函数以0为界，输出从0切换为1(或者从1切换为0)


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

# 广播
t = np.array([1.0, 2.0, 3.0])
print(1 + t)
print(1 / t)

# 画出sigmoid函数
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # y轴标签
plt.title("sigmoid")  # 标题
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.xticks(np.arange(-6, 6.1, 1))  # 设置x轴刻度
plt.show()


# 将两个函数画在一起
x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1, label="step_function")
plt.plot(x, y2, linestyle="--", label="sigmoid")  # 用虚线绘制
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # y轴标签
plt.title("step_function & sigmoid")  # 标题
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.xticks(np.arange(-6, 6.1, 1))  # 设置x轴刻度
plt.show()
# 相对于阶跃函数只能返回0或1，sigmoid函数可以返回0.731等实数
# 感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号


# ReLU函数(Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # y轴标签
plt.title("relu")  # 标题
plt.ylim(-1, 5.5)  # 指定y轴的范围
plt.xticks(np.arange(-6, 6.1, 1))  # 设置x轴刻度
plt.show()


# 多维数组运算
# 一维数组(array)
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))  # 数组维数
print(A.shape)  # 数组形状
print(A.shape[0])

# 二维矩阵(matrix)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))  # 数组维数
print(B.shape)  # 数组形状
# 数组的横向排列称为行(row)，纵向排列称为列(column)

# 矩阵乘法
# 2*2 2*2 = 2*2
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)

print(np.dot(A, B))

# 2*3 3*2 = 2*2
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

print(np.dot(A, B))
# 在矩阵的乘积运算中，对应维度的元素个数要保持一致

# 3*2 2 = 3
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)
B = np.array([7, 8])
print(B.shape)

print(np.dot(A, B))


# 神经网络的内积
X = np.array([1, 2])
print(X)
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
Y = np.dot(X, W)
print(Y)


# 三层神经网络的实现
# start
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


# 恒等函数
def identity_function(x):
    return x


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [0.31682708 0.69627909]
# end

# softmax函数
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)  # 指数函数
print(exp_a)
sum_exp_a = np.sum(exp_a)  # 指数函数的和
print(sum_exp_a)
y = exp_a / sum_exp_a
print(y)


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# 解决溢出问题
a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a)))  # softmax函数运算
c = np.max(a)
print(a - c)
print(np.exp(a - c) / np.sum(np.exp(a - c)))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# softmax函数的输出是0.0到1.0之间的实数且softmax函数的输出值的总和是1
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
