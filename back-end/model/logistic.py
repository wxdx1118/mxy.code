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



class LogisticRegression():
    """逻辑回归类"""
    def __init__(self, n):
        self.STOP_ITER = 0#迭代次数
        self.STOP_COST = 1#损失值 L(w)
        self.STOP_GRAD = 2#梯度变化
        self.n = n
    
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
            if k >= self.n: 
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

    def predict(self, X, theta):
        return [1 if x >= 0.5 else 0 for x in self.model(X, theta)]
    
    def runExpe(self,data, theta, batchSize, stopType, thresh, alpha):

        theta, iter, costs, grad, dur = self.descent(data, theta, batchSize, stopType, thresh, alpha)
        name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
        name += f" data / learning rate: {alpha} / "
        if batchSize==1:  
            strDescType = "Stochastic"
        elif batchSize==65: 
            strDescType = "Gradient"
        else: 
            strDescType = f"Mini-batch ({batchSize})"
        name += strDescType + " descent / Stop: "
        if stopType == self.STOP_ITER: 
            strStop = f"{thresh} iterations"
        elif stopType == self.STOP_COST: 
            strStop = f"costs change < {thresh}"
        else: 
            strStop = f"gradient norm < {thresh}"
        name += strStop
    #     print(name)
        print (f"Iter: {iter} / Last cost: {costs[-1]:03.2f} / Duration: {dur:03.2f}s")
        plt.subplots(figsize=(12,4))
        plt.plot(np.arange(len(costs)), costs, 'r')
        plt.xlabel('Iterations')
        plt.ylabel('Cost') 
        plt.xlim(-1,)
        plt.title(name.upper())
        return theta


    #模型评估
    def Assess():
        pass

    #调参
    def AdParam():
        pass
