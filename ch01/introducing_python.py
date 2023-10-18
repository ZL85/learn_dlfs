# 算数计算
print("1 + 1 =", 1 + 1)
print("1 - 1 =", 1 - 1)
print("2 * 2 =", 2 * 2)
print("4 / 2 =", 4 / 2)
print("9 // 2 =", 9 // 2)  # 整除
print("2 ** 4 =", 2**4)  # 幂


# 数据类型(data type)
print(type(10))
print(type(3.14))
print(type("hello"))


# 变量
x = 100
print(x)
y = 3.14
print(x * y)
print(type(x * y))


# 列表
a = [1, 2, 3, 4, 5]
print(a)
print(len(a))
print(a[0], a[-1])
a[-2] = 66
print(a[3])
print(a)
print(a[0:2])
print(a[1:])
print(a[:3])

# 字典
me = {"height": 180}
print(me["height"])
me["weight"] = 70
print(me)


# 布尔型
hungry = True
sleepy = False
print(type(hungry))
print(type(sleepy))
print(not hungry)
print(not sleepy)
print(hungry and sleepy)
print(hungry or sleepy)


# if语句
if hungry:
    print("I'm hungry!")
else:
    print("I'm not hungry!")


# for语句
for item in range(0, 10, 2):
    print(item)


# 函数
def hello():
    print("hello world!")


hello()


def hello(obj):
    print("hello " + obj + "!")


hello("ahu")


# 类
# class 类名:
#     def __init__(self, 参数, 参数, ...):  #构造方法
#         pass
#     def 方法1名(self, 参数, 参数, ...):  #方法1
#         psss
#     def 方法1名(self, 参数, 参数, ...):  #方法2
#         psss
class Man:
    def __init__(self, name) -> None:
        self.name = name
        print("Initialized!")

    def hello(self):
        print("hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")


m = Man("zl")
m.hello()
m.goodbye()


# Numpy
# 导入外部库
import numpy as np

# 生成numpy数组
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

# numpy的算术运算
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)  # element-wise product
print(x / y)

print(x / 2.0)  # 广播

# numpy的n维数组
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)  # 形状
print(A.dtype)  # 数组类型

# 矩阵算术运算
B = np.array([[3, 0], [6, 0]])
print(A + B)
print(A * B)

print(A)
print(A * 10)

# 广播(broadcast)
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])  # 第0行
print(X[0][1])  # (0,1)的元素

for row in X:
    print(row)

X = X.flatten()  # 将X转换为一维数组
print(X)
print(X[np.array([0, 2, 4])])
print(X > 15)  # 获取满足一定条件的元素
print(X[X > 15])


# Matplotlib(可视化)
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)  # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")  # 用虚线绘制
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # x轴标签
plt.title("sin & cos")  # 标题
plt.xlim((0, 6.1))  # 设置x坐标轴范围
plt.ylim((-1, 1))  # 设置y坐标轴范围
plt.xticks(np.arange(0, 6.1, 0.5))  # 设置x轴刻度
plt.legend()
plt.show()


# 显示图像
from matplotlib.image import imread

img = imread("../dataset/lena.png")
plt.imshow(img)

plt.show()
