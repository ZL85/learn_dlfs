# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


# 通过不断地沿梯度方向前进，逐渐减小函数值的过程就是梯度法（gradient method）
# 寻找最小值的梯度法称为梯度下降法（gradient descent method）
# 寻找最大值的梯度法称为梯度上升法（gradient ascent method）
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    f是要进行最优化的函数
    init_x是初始值
    lr是学习率learning rate
    step_num是梯度法的重复次数
    """
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot([-5, 5], [0, 0], "--b")
plt.plot([0, 0], [-5, 5], "--b")
plt.plot(x_history[:, 0], x_history[:, 1], "o")

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
