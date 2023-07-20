from sklearn.metrics import recall_score


class recall:
    def __init__(self) -> None:
        self.name='Recall'

    def compute(self, y_pred, y_test):
        # 计算召回率
        recall = recall_score(y_test, y_pred, average='macro')
        return recall
        #print("Recall: {:.2f}%".format(recall * 100))
