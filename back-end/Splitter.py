from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

#拆分数据集
class Splitter:
    def __init__(self,data) -> None:
        super().__init__()
        self.data=data
        self.rows=data.shape[1]-1
        self.X=None
        self.y=None
    
    def split(self, test_ratio):
        #拆分特征值与标签值
        self.X = self.data.iloc[:,0:self.rows]
        self.y = self.data.iloc[:,self.rows]
        #划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_ratio, random_state=42)

        return X_train, X_test, y_train, y_test
