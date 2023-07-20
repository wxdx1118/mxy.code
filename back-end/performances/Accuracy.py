from sklearn.metrics import accuracy_score


class accuracy:
    def __init__(self) -> None:
        self.name='Accuracy'
        
    def compute(self, y_pred, y_test):
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

        #print("Accuracy: {:.2f}%".format(accuracy * 100))