# coding=utf8
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter
import numpy as np

class KNearestNeighbors:
    parameter={
        'k 最近邻居数':2
    }

    def __init__(self,learning_rate):
        self.k = int(self.parameter['k 最近邻居数'])
        self.learning_rate = learning_rate
        self.X_train = None
        self.y_train = None
    
    @classmethod
    def set_parameter(cls,dic):
        cls.parameter=dic


    def train(self, X_train, y_train):
        # 存储训练集数据和标签
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        # 计算欧氏距离
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        y_pred = []
        for test_sample in X_test:
            distances = []
            for train_sample, label in zip(self.X_train, self.y_train):
                
                # 计算测试样本与训练样本之间的距离，并存储距离和对应的标签
                distance = self.euclidean_distance(test_sample, train_sample) * self.learning_rate
                distances.append((distance, label))

            # 根据距离进行排序，选择最近的k个样本
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            labels = [label for distance, label in k_nearest]

            # 统计k个样本中标签出现最多的标签作为预测结果
            most_common = Counter(labels).most_common()
            predicted_label = most_common[0][0]
            y_pred.append(predicted_label)

        return np.array(y_pred)
    
    def Plot_Scatter(self, X_test):
        # 预测分类散点图
        # 绘制散点图
        y_pred=self.predict( X_test)
        
        x = []
        y = []

        x=X_test.iloc[:,0]
        y=X_test.iloc[:,1]
      

        # 绘制散点图
        plt.scatter(x, y, c=y_pred)
        # 设置坐标轴标签
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])

        # 显示图形
        plt.show()



        
