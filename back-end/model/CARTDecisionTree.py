import numpy as np
from model.Model import Model

class Node:
    def __init__(self, feature=None, threshold=None, label=None, value=None):
        self.feature = feature  # 分裂特征索引
        self.threshold = threshold  # 分裂阈值
        self.label = label  # 叶节点标签（分类问题）
        self.value = value  # 叶节点预测值（回归问题）
        self.left = None  # 左子节点
        self.right = None  # 右子节点

class CARTDecisionTree(Model):
    parameter={
        'max_depth 最大深度':100,
        'pruning 是否进行剪枝:True/False':True
    }
    
    def __init__(self,learning_rate=1.0):
        self.max_depth = int(self.parameter['max_depth 最大深度'])  # 最大深度限制
        self.learning_rate = learning_rate  # 学习率参数
        self.pruning = self.parameter['pruning 是否进行剪枝:True/False']  # 是否进行剪枝
        self.root = None  # 决策树根节点

    # 用于随机森林
    def __init__(self,learning_rate=1.0,max_depth=20):
        self.max_depth = max_depth # 最大深度限制
        self.learning_rate = learning_rate  # 学习率参数
        self.pruning = None # 是否进行剪枝
        self.root = None  # 决策树根节点

    @classmethod # 使用类方法装饰器
    def set_parameter(cls, dic):
        cls.parameter = dic

    # 分类问题 计算gini指数
    def _calc_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        gini = 1.0 - np.sum((counts / np.sum(counts))**2)
        return gini

    # 回归问题 计算均方误差
    def _calc_mse(self, y):
        return np.mean((y - np.mean(y))**2)

    # 数据集切分
    def _split_dataset(self, X, y, feature, threshold):
        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    # 寻找最优特征及切分点
    def _find_best_split(self, X, y):
        best_gini = np.inf
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature, threshold)
                gini = len(y_left) / n_samples * self._calc_gini(y_left) + len(y_right) / n_samples * self._calc_gini(y_right)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # 构造决策树
    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            if self.pruning:
                if self.loss_func == 'gini':
                    value = np.argmax(np.bincount(y))
                elif self.loss_func == 'mse':
                    value = np.mean(y)
            else:
                if self.loss_func == 'gini':
                    value, _ = np.unique(y, return_counts=True)
                    value = value[0]
                elif self.loss_func == 'mse':
                    value = np.mean(y)

            return Node(label=value, value=value)

        feature, threshold = self._find_best_split(X, y)
        X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature, threshold)

        node = Node(feature=feature, threshold=threshold)
        node.left = self._build_tree(X_left, y_left, depth+1)
        node.right = self._build_tree(X_right, y_right, depth+1)

        return node

    # 后剪枝
    def _prune(self, node, X, y):
        if node.left and node.right:
            X_left, y_left, X_right, y_right = self._split_dataset(X, y, node.feature, node.threshold)
            if len(y_left) > 0 and len(y_right) > 0:
                node.left = self._prune(node.left, X_left, y_left)
                node.right = self._prune(node.right, X_right, y_right)

        if not node.left and not node.right:
            return node

        if self.loss_func == 'gini':
            error_node = len(y) / (len(y) + len(node.left.label) + len(node.right.label))
            error_merge = len(y) / len(np.argmax(np.bincount(y)))
        elif self.loss_func == 'mse':
            error_node = np.sum((y - np.mean(y))**2)
            if node.value is not None:
                error_merge = np.sum((y - node.value)**2)
            else:
                error_merge = np.inf

        if error_merge <= error_node * (len(y) - 1):
            if self.loss_func == 'gini':
                return Node(label=np.argmax(np.bincount(y)))
            elif self.loss_func == 'mse':
                return Node(value=np.mean(y))

        return node

    # 对单个样本进行预测
    def _predict_sample(self, x, node):
        if node.label is not None:
            return node.label
        elif node.value is not None:
            return node.value
        else:
            if x[node.feature] <= node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)

    # 训练模型
    def train(self, X_train, y_train):
        #X_train=X_train.values
        #y_train=y_train.values
        if np.unique(y_train).dtype == object:
            self.loss_func = 'gini'
        else:
            self.loss_func = 'mse'

        self.root = self._build_tree(X_train, y_train)

        if self.pruning:
            self.root = self._prune(self.root, X_train, y_train)

    def predict(self, X_test):
        #X_test=X_test.values
        y_pred = []

        for x in X_test:
            y_pred.append(self._predict_sample(x, self.root))

        return np.array(y_pred)
