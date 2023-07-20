import pandas as pd
import numpy as np

# 定义多元回归线性模型类
class LinearRegression:

    parameter={
        'num_iter 迭代次数':10,
    }
    
    def __init__(self, learning_rate):
        self.lr = learning_rate  # 学习率
        self.num_iter = int(self.parameter['num_iter 迭代次数'] ) # 迭代次数
    
    @classmethod # 使用类方法装饰器
    def set_parameter(cls, dic):
        cls.parameter = dic

    def train(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1) # 重组成二维数组
        self.n_samples, self.n_features = self.X.shape
        self.weights = np.zeros((self.n_features, 1))  # 初始化权重(np.zeros:返回来一个给定形状和类型的用0填充的数组)
        self.bias = 0  # 初始化偏置项
        
        # 梯度下降迭代更新权重和偏置项
        for _ in range(self.num_iter):
            y_pred = self.predict(self.X)
            dw = (1 / self.n_samples) * np.dot(self.X.T, (y_pred - self.y))  # 梯度下降算法中权重w的偏导数（self.X.T表示特征矩阵X的转置；np.dot(self.X.T, (y_pred - self.y))表示特征矩阵X与预测值与真实值之间的误差的内积）
            db = (1 / self.n_samples) * np.sum(y_pred - self.y)  # 偏置项b的偏导数（np.sum(y_pred - self.y)表示预测值与真实值之间的误差之和）
            self.weights -= self.lr * dw  # 更新
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias  # 线性回归(np.dot矩阵乘法函数)
