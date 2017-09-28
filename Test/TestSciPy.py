import numpy as np
from scipy.optimize import leastsq
import pylab as pl


def func(x, p):  # 这个函数就是我们要拟合的函数

    A, k, theta = p
    return A * np.sin(2 * np.pi * k * x + theta)


def errors(p, y, x):  # 这个定义的是拟合的原函数的误差值
    return y - func(x, p)


x = np.linspace(0, -2 * np.pi, 100)  # numpy中的linspace产生[0,-2pi]中的100个数（线性的）
A, k, theta = [10, 0.34, np.pi / 6]  # 原函数的三个参数
y0 = func(x, [A, k, theta])  # 真实的数据集
y1 = y0 + 2 * np.random.randn(len(x))  # 加入噪声的数据集
p0 = [2, 0.2, np.pi]
plsq = leastsq(errors, p0, args=(y1, x))  # 通过优化函数leastsq进行迭代优化

print("real parameters:", [A, k, theta])
print("matching parameters:", plsq[0])

pl.plot(x, y0, 'bo', label="real data")
pl.plot(x, y1, 'r1', label="matching data")
pl.plot(x, func(x, plsq[0]), label='matching curve')
pl.legend()
pl.show()  # 可视化
