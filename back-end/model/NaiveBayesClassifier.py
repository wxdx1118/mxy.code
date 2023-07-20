import numpy as np
from model.Model import Model


class NaiveBayesClassifier(Model):
    parameter={}

    def __init__(self, learning_rate=1.0):
        self.alpha = learning_rate  # 学习率参数
        self.priors = None  # 先验概率
        self.means = None  # 特征均值
        self.stds = None  # 特征标准差

    @classmethod # 使用类方法装饰器
    def set_parameter(cls, dic):
        cls.parameter = dic

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)  # 所有类别
        self.n_classes = len(self.classes)  # 类别数量
        self.n_features = X_train.shape[1]  # 特征数量

        self.priors = np.zeros(self.n_classes)  # 先验概率
        self.means = np.zeros((self.n_classes, self.n_features))  # 特征均值
        self.stds = np.zeros((self.n_classes, self.n_features))  # 特征标准差

        for label in self.classes:
            X_label = X_train[y_train == label]
            self.priors[label] = len(X_label) / len(X_train)
            self.means[label] = X_label.mean(axis=0)
            self.stds[label] = X_label.std(axis=0)

    def predict(self, X_test):
        y_pred = []

        for sample in X_test:
            likelihood = np.zeros(self.n_classes)
            for label in self.classes:
                prob = 1
                for feature in range(self.n_features):
                    exponent = np.exp(-((sample[feature] - self.means[label][feature]) ** 2) / (2 * (self.stds[label][feature] ** 2)))
                    prob *= exponent / (np.sqrt(2 * np.pi) * self.stds[label][feature])
                likelihood[label] = prob

            posterior = self.priors * likelihood
            y_pred.append(np.argmax(posterior))

        return np.array(y_pred)
    
