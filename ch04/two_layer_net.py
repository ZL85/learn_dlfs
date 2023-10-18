# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np


class TwoLayerNet:
    # 进行初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        input_size: 输入层的神经元数
        hidden_size: 隐藏层的神经元数
        output_size: 输出层的神经元数
        """
        # 初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        # 保存神经网络的参数的字典型变量（实例变量）
        # params['W1']是第1 层的权重，params['b1']是第1 层的偏置
        # params['W2']是第2 层的权重，params['b2']是第2 层的偏置

    # 进行识别（推理）
    def predict(self, x):
        """
        x是图像数据
        """
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 计算损失函数的值
    def loss(self, x, t):
        """
        x:输入数据
        t:监督数据
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 计算权重参数的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    # numerical_gradient()的高速版
    def gradient(self, x, t):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)

        return grads


# 保存梯度的字典型变量（numerical_gradient()方法的返回值）
# grads['W1']是第1 层权重的梯度，grads['b1']是第1 层偏置的梯度
# grads['W2']是第2 层权重的梯度，grads['b2']是第2 层偏置的梯度
