from sklearn.metrics import r2_score


class R2_score:
    def __init__(self) -> None:
       self.name='R2_score'

    def compute(self, y_pred, y_test):
        # 计算判定系数
        r2 = r2_score(y_test, y_pred)
        return r2
        #print("R2 Score: {:.2f}".format(r2))
