import numpy as np
from model.Model import Model
from model.CARTDecisionTree import CARTDecisionTree

class RandomForest(Model):
    parameter={
        'max_depth 决策树深度':20,
        'n_trees 决策树颗数':10
    }

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        self.max_depth = int(self.parameter['max_depth 决策树深度']) # 决策树深度
        self.n_trees = int(self.parameter['n_trees 决策树颗数'])    # 决策树个数
        self.trees = []

    @classmethod # 使用类方法装饰器
    def set_parameter(cls, dic):
        cls.parameter = dic

    def train(self, X_train, y_train):
        for _ in range(self.n_trees):
            tree = CARTDecisionTree(max_depth=self.max_depth,learning_rate=self.learning_rate)
            tree.train(X_train, y_train)
            self.trees.append(tree)

    def predict(self, X_test):
        predictions = np.zeros(len(X_test))
        for tree in self.trees:
           predictions += self.learning_rate * tree.predict(X_test) 
        return np.round(predictions / len(self.trees))

