from sklearn.metrics import mean_absolute_error
import numpy as np

class mae:
    def __init__(self) -> None:
        self.name='Mae'
    
    def compute(self, y_pred, y_test):
        # 计算平均绝对误差
        mae = mean_absolute_error(y_test, y_pred)
        return mae
       # print("MAE: {:.2f}".format(mae))