from sklearn.metrics import precision_score


class precision:
    def __init__(self) -> None:
        self.name='Precision'

    def compute(self, y_pred, y_test):
        # 计算精确度
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        return precision
        #print("Precision: {:.2f}%".format(precision * 100))