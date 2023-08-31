#导入数据集 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.Dataset import Dataset

class DigitsDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)
               
    def load(self):
        df=super().load()
        df=self.basic(df)
        self.insertOne(df)
        df=self.toDichotomy(df)
        return df

    def insertOne(self,df):#插入一列ones
        df.insert(0, 'Ones', 1) # 引入截距项
    
    def toDichotomy(self,df):#转成二分类
        df.iloc[:,-1]=(df.iloc[:,-1] > 4).astype(np.int)
        return df
    
    def toNumpy(self):#转换成数组
        return self.data.values
    

