from sklearn.metrics import explained_variance_score

class variance:
    def __init__(self) -> None:
        self.name='variance'

    def compute(self, y_pred, y_test):
        # 计算方差
        variance = explained_variance_score(y_test, y_pred)
        return variance
        #print("variance: {:.2f}".format(variance))