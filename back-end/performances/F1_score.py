from sklearn.metrics import f1_score

class F1_score:
    def __init__(self) -> None:
        self.name='F1-socre'

    def compute(self, y_pred, y_test):
        # 计算F1分数
        f1 = f1_score(y_test, y_pred, average='macro')
        return f1
        #print("F1 Score: {:.2f}%".format(f1 * 100))