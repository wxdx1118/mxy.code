import time
from sklearn.datasets import load_digits
##用于可视化图表
import matplotlib.pyplot as plt
##用于做科学计算
import numpy as np
##用于做数据分析
import pandas as pd
##用于加载数据或生成数据等
from sklearn import datasets
##加载线性模型
from sklearn import linear_model
###用于交叉验证以及训练集和测试集的划分
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
#from sklearn.cross_validation import cross_val_score
###这个模块中含有评分函数，性能度量，距离计算等
from sklearn import metrics
###用于做数据预处理
from sklearn import preprocessing


'''class LogisticRegresion():
    parameter={
        "batchSize批量大小":200,
        "stopType停止类型(0/1/2)":0,
        "thresh停止标准阈值":5000
    }
    @classmethod
    def set_parameter(cls,dic):
        cls.parameter=dic
    
    """逻辑回归类"""
    def __init__(self, learning_rate):
        self.STOP_ITER = 0  # 迭代次数
        self.STOP_COST = 1  # 损失值 L(w)
        self.STOP_GRAD = 2  # 梯度变化
        self.batchSize = int(self.parameter['batchSize批量大小'])  
        self.learning_rate = learning_rate 
        self.stopType = self.parameter['stopType停止类型(0/1/2)']  
        self.thresh = self.parameter['thresh停止标准阈值']  
        self.theta = None
        self.xmodel=None
  

    def sigmoid(self, z):
        """
            sigmoid函数
            将预测值映射成概率
        """
        return 1 / (1 + np.exp(-z))

    def model(self, X, theta):
        """
            预测函数：返回预测值 h
        """
        return self.sigmoid(np.dot(X, theta.T))

    def cost(self, X, y, theta):
        """损失函数 L(w)"""  
        left = np.multiply(-y, np.log(self.model(X, theta)))
        right = np.multiply(1 - y, np.log(1 - self.model(X, theta)))
        return np.sum(left - right) / (len(X))

    def gradient(self, X, y, theta):
        """计算梯度"""
        grad = np.zeros(theta.shape)
        error = (self.model(X, theta)- y).ravel()
        for j in range(len(theta.ravel())): #for each parmeter
            term = np.multiply(error, X[:,j])
            grad[0, j] = np.sum(term) / len(X)
        return grad

    def stopCriterion(self, type, value, threshold):
        """
            停止标准函数：
                1.迭代次数
                2.损失值变化
                3.梯度变化
        """
        if type == self.STOP_ITER:        
            return value > threshold
        elif type == self.STOP_COST:      
            return abs(value[-1]-value[-2]) < threshold
        elif type == self.STOP_GRAD:      
            return np.linalg.norm(value) < threshold

    def shuffleData(self, data):
        """洗牌"""
        np.random.shuffle(data)
        cols = data.shape[1]
        X = data[:, 0:cols-1]
        y = data[:, cols-1:]
        return X, y

    def descent(self, data, theta, batchSize, stopType, thresh, alpha):
        """梯度下降求解"""
        init_time = time.time()
        i = 0 # 迭代次数
        k = 0 # batch
        X, y = self.shuffleData(data)
        grad = np.zeros(theta.shape) # 计算的梯度
        costs = [self.cost(X, y, theta)] # 损失值

        while True:
            grad = self.gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
            k += batchSize #取batch数量个数据
            if k >= len(X): 
                k = 0 
                X, y = self.shuffleData(data) #重新洗牌
            theta = theta - alpha*grad # 参数更新
            costs.append(self.cost(X, y, theta)) # 计算新的损失
            i += 1 

            if stopType == self.STOP_ITER:       
                value = i
            elif stopType == self.STOP_COST:     
                value = costs
            elif stopType == self.STOP_GRAD:     
                value = grad
            if self.stopCriterion(stopType, value, thresh): 
                break

        return theta, i-1, costs, grad, time.time() - init_time

    def train(self, X_train,y_train):
        """
        训练模型
        :param batchSize: 批量大小
        :param stopType: 停止标准类型
        :param thresh: 停止标准阈值
        :param alpha: 学习率
        :return: 训练后的参数、迭代次数、损失值列表、梯度、训练耗时
        """
       
        n_features = X_train.shape[1]
        batchSize=self.batchSize
        stopType=self.stopType
        thresh=self.thresh
        alpha=self.learning_rate
        init_theta=np.zeros((1, n_features))
        self.theta, iter, costs, grad, dur = self.descent(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1), init_theta,batchSize, stopType, thresh, alpha)
        return self.theta, iter, costs, grad, dur
       
        # 创建逻辑回归模型对象
        model =linear_model.LogisticRegression()
        # 使用训练数据拟合模型
        model.fit(X_train, y_train)
        # 返回训练好的模型
        self.theta= np.array([model.intercept_] + model.coef_[0].tolist())
        return model

    def predict(self, X_test):
        """
        预测函数
        :param X_test: 输入特征
        :return: 预测结果
        """
        if self.theta is None:
            raise ValueError("Model not trained yet. Please call 'train' function first.")
        return [1 if x >= 0.5 else 0 for x in self.model(X_test, self.theta)]
'''
import numpy as np

class LogisticRegression:
    parameter={
        "num_iterations迭代次数":1000
    }
    @classmethod
    def set_parameter(cls,dic):
        cls.parameter=dic
    
    def __init__(self,learning_rate=0.01):
        self.theta = None
        self.learning_rate=learning_rate
        self.num_iterations=int(self.parameter['num_iterations迭代次数'])  
    
    def train(self, X_train, y_train):
        # 在特征矩阵 X_train 前添加一列全为1的偏置项
        X_train = np.insert(X_train, 0, 1, axis=1)
        
        # 初始化参数 theta
        num_features = X_train.shape[1]
        self.theta = np.zeros(num_features)
        learning_rate=self.learning_rate
        num_iterations=self.num_iterations
        # 梯度下降训练模型
        for _ in range(num_iterations):
            theta_grad = np.dot(X_train.T, self._sigmoid(np.dot(X_train, self.theta)) - y_train)
            self.theta -= learning_rate * theta_grad
    
    def predict(self, X_test):
        # 在特征矩阵 X_test 前添加一列全为1的偏置项
        X_test = np.insert(X_test, 0, 1, axis=1)
        
        # 预测类别
        probabilities = self._sigmoid(np.dot(X_test, self.theta))
        predictions = (probabilities >= 0.5).astype(int)
        
        return predictions
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
       

