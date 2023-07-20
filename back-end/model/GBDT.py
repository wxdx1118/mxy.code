import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import log_loss

class GBDTRegressor:
    parameter={
        'n_estimators 决策树棵树': 0 ,
        'max_depth 决策树最大深度': 0
    }
    def __init__(self, learning_rate=0.1):
        self.n_estimators = int(self.parameter['n_estimators 决策树棵树'] ) # 弱学习器数量
        self.learning_rate = learning_rate  # 学习率
        self.max_depth = int(self.parameter['max_depth 决策树最大深度'] ) # 决策树最大深度
        
        self.estimators = []  # 弱学习器列表
    
    @classmethod
    def set_parameter(cls,dic):
        cls.parameter=dic

    def _calc_residual(self, y_true, y_pred):#计算残差
        return y_true - y_pred
    
    def _calc_gradient(self, y_true, y_pred):#计算梯度
        # 在回归问题中，梯度即为残差
        return self._calc_residual(y_true, y_pred)
    
    def _fit_estimator(self, X, y):
        # 构建并训练一个决策树作为弱学习器
        estimator = DecisionTreeRegressor(max_depth=self.max_depth)
        estimator.fit(X, y)
        
        return estimator
    
    def train(self, X, y):
        # 初始化预测值为全零数组
        y_pred = np.zeros_like(y)
        
        # 逐个训练弱学习器
        for _ in range(self.n_estimators):
            gradient = self._calc_gradient(y, y_pred)
            
            # 使用负梯度作为目标值训练弱学习器
            estimator = self._fit_estimator(X, gradient)
            
            # 更新预测值
            y_pred += self.learning_rate * estimator.predict(X)
            
            # 将训练好的弱学习器添加到学习器列表中
            self.estimators.append(estimator)
    
    def predict(self, X):
        # 预测新样本的输出值
        y_pred = np.zeros(len(X))
        
        # 逐个弱学习器进行预测并累加结果
        for estimator in self.estimators:
            y_pred += self.learning_rate * estimator.predict(X)
        
        return y_pred

class GBDTClassifier:
    parameter={
        'n_estimators 决策树棵树': 0 ,
        'max_depth 决策树最大深度': 0
    }

    #构造函数 包含参数：决策树个数、学习率、最大树深度
    def __init__(self,  learning_rate=0.1):
        self.n_estimators = int(self.parameter['n_estimators 决策树棵树'])
        self.learning_rate = learning_rate
        self.max_depth =int( self.parameter['max_depth 决策树最大深度'])
        self.estimators = []
        self.init_pred = None
    
    @classmethod  # 使用类方法装饰器
    def set_parameter(cls, dic):
        cls.parameter = dic
    
    #预测值转换为概率
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #计算损失函数的梯度 概率残差
    def gradient(self, y, y_pred):
        return y - self.sigmoid(y_pred)
    
    #更新叶子节点的值以最小化损失函数
    def update_terminal_region(self, tree, X, y, residual):
        terminal_regions = tree.apply(X)
        for leaf in np.unique(terminal_regions):
            mask = (terminal_regions == leaf)
            numerator = np.sum(y[mask]) - 0.5
            denominator = np.sum(np.abs(0.5 - y[mask]) * (1 - y[mask])) + 1e-16
            leaf_value = np.log(numerator / denominator) if numerator * denominator > 0 else 0
            tree.tree_.value[leaf][0][0] = leaf_value
    
    #训练GBDT分类器
    def train(self, X, y):
        y_pred = np.zeros_like(y).astype(float)
        self.init_pred = np.log(np.mean(y) / (1 - np.mean(y)))
        
        for i in range(self.n_estimators):
            residual = self.gradient(y, y_pred)
            #print("第",i,"棵树：",residual)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            
            self.update_terminal_region(tree, X, y, residual)
            
            self.estimators.append(tree)
            y_pred += self.learning_rate * tree.predict(X) + self.init_pred
    
    #返回预测样本属于各个类别的概率
    def predict_proba(self, X):
        pred = np.zeros((X.shape[0],))
        
        for tree in self.estimators:
            pred += self.learning_rate * tree.predict(X) + self.init_pred
        
        prob = self.sigmoid(pred)
        return np.c_[1 - prob, prob]
    
    #根据预测概率返回类别
    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)
