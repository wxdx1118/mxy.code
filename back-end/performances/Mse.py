from sklearn.metrics import mean_squared_error


class mse:
    def __init__(self) -> None:
        self.name='Mse'
        
    def compute(self, y_pred, y_test):
        # 计算均方差
        mse = mean_squared_error(y_test, y_pred)
        return mse
        #print("MSE: {:.2f}".format(mse))