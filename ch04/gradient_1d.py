# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


# 数值微分(numerical differentiation)
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


# y = 0.01x^2 + 0.1x
def function_1(x):
    return 0.01 * x**2 + 0.1 * x


# 切线
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

tf1 = tangent_line(function_1, 10)
y3 = tf1(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)

plt.xlim((0, 20.1))  # 设置x坐标轴范围
plt.ylim((-1, 6.1))  # 设置y坐标轴范围
plt.xticks(np.arange(0, 20.1, 2.5))  # 设置x轴刻度
plt.show()
