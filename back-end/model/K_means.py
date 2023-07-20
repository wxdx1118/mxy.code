from matplotlib import pyplot as plt
import numpy as np

class KMeans:
    parameter={
        'n_clusters 聚类的数量':2,
        'max_iter 最大迭代次数':100
    }

    def __init__(self,learning_rate):
        self.n_clusters = int(self.parameter['n_clusters 聚类的数量'] ) # 聚类的数量
        self.max_iter = int(self.parameter['max_iter 最大迭代次数'] ) # 最大迭代次数
        self.learning_rate = learning_rate  # 学习率
        self.centroids = None  # 聚类中心
    
    @classmethod
    def set_parameter(cls,dic):
        cls.parameter=dic

    def train(self, X_train,y_train):
        y_train=y_train
        # 初始化聚类中心
        np.random.seed(0)
        idx = np.random.choice(range(X_train.shape[0]), size=self.n_clusters, replace=False)
        self.centroids = X_train[idx]

        # 迭代更新聚类中心
        for _ in range(self.max_iter):
            distances = self.calculate_distances(X_train)  # 计算每个样本到聚类中心的距离
            labels = self.assign_labels(distances)  # 分配样本到最近的聚类中心
            new_centroids = self.update_centroids(X_train, labels)  # 更新聚类中心

            # 判断聚类中心是否收敛
            if np.all(self.centroids == new_centroids):
                break

            # 使用学习率进行更新
            self.centroids = self.centroids + self.learning_rate * (new_centroids - self.centroids)

    def calculate_distances(self, X_train):
        """
        计算每个训练样本到聚类中心的距离。

        参数：
        - X_train:训练数据集，形状为 (样本数量, 特征数量)

        返回：
        - distances:每个训练样本到聚类中心的距离，形状为 (样本数量, 聚类数量)
        """
        distances = np.zeros((X_train.shape[0], self.n_clusters))  # 初始化距离矩阵

        for i in range(self.n_clusters):
            centroid = self.centroids[i]  # 获取第 i 个聚类中心坐标
            distances[:, i] = np.linalg.norm(X_train - centroid, axis=1)  # 计算每个训练样本到聚类中心的距离

        return distances

    def assign_labels(self, distances):
        """
        分配聚类标签：根据每个样本到聚类中心的距离，分配最近的聚类标签。
        参数：
        - distances:每个样本到聚类中心的距离，形状为 (样本数量, 聚类数量)
        返回：
        - labels:每个样本的聚类标签，形状为 (样本数量,)
        """
        return np.argmin(distances, axis=1)  # 返回每个样本距离最近聚类中心的索引作为聚类标签

    def update_centroids(self, X_train, labels):
        """
        更新聚类中心：根据当前样本的标签，计算每个聚类的新中心点坐标。
        参数：
        - X_train:训练数据集，形状为 (样本数量, 特征数量)
        - labels:每个样本的聚类标签，形状为 (样本数量,)
        返回：
        - new_centroids:更新后的聚类中心，形状为 (聚类数量, 特征数量)
        """
        new_centroids = np.zeros((self.n_clusters, X_train.shape[1]))  # 初始化新的聚类中心

        for i in range(self.n_clusters):
            cluster_points = X_train[labels == i]  # 获取属于第 i 个聚类的所有样本
            if cluster_points.size > 0:  # 如果该聚类有样本
                new_centroids[i] = np.mean(cluster_points, axis=0)  # 计算该聚类的新中心点坐标

        return new_centroids

    def predict(self, X_test):
        """
        对测试数据进行聚类预测。
        参数：
        - X_test:测试数据集，形状为 (样本数量, 特征数量)
        返回：
        - labels:每个测试样本的聚类标签，形状为 (样本数量,)
        """
        distances = self.calculate_distances(X_test)  # 计算测试样本到聚类中心的距离
        labels = self.assign_labels(distances)  # 分配每个样本到最近的聚类中心
        return labels

    # def plot_elbow(self, X_train):
    #     """
    #     绘制肘部法则图形。在不同聚类数量下,计算并绘制误差平方和(SSE)。
    #     参数：
    #     - X_train:训练数据集,形状为 (样本数量, 特征数量)
    #     """
    #     sse = []  # 存储不同聚类数量下的 SSE 值

    #     for k in range(1, self.n_clusters + 1):
    #         self.n_clusters = k  # 设置当前聚类数量
    #         self.fit(X_train)  # 对训练数据进行聚类
    #         distances = self.calculate_distances(X_train)  # 计算训练样本到聚类中心的距离
    #         labels = self.assign_labels(distances)  # 分配每个样本到最近的聚类中心
    #         sse.append(np.sum(np.min(distances, axis=1)) / X_train.shape[0])  # 计算当前聚类数量下的 SSE 并存储

    #     plt.plot(range(1, self.n_clusters + 1), sse, marker='o')  # 绘制 SSE 曲线
    #     plt.xlabel('Number of clusters')  # x 轴标签
    #     plt.ylabel('SSE')  # y 轴标签
    #     plt.title('Elbow Method')  # 图形标题
    #     plt.show()  # 显示图形
