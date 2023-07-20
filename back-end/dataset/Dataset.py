import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # 标准化

class Dataset:
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.data = None
        
    def load(self):
        self.data=pd.read_csv(self.path)
        return self.data
    
    #基本数据预处理
    def basic(self,df):
        #查看有无缺失值
        missing_count = df.isnull().sum().sum()
        #print(missing_count)
        #缺失值处理 linear：线性插值
        #print(df.info())
        df = df.interpolate(method='linear', limit_direction='forward')
        df = df.interpolate(method='linear', limit_direction='backward')

        #重复值
        df.drop_duplicates()
        return df

    #归一化
    def preprocessing(self,df):
        # 获取特征列
        feature_cols = df.columns[:-1]
        # 仅对特征值归一化
        stand = StandardScaler()#标准化操作
        #计算训练数据的均值和方差，把数据转换成标准的正态分布
        df[feature_cols]  = stand.fit_transform(df[feature_cols])

        return df

