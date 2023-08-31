# 导入数据集划分工具
from splitter.Splitter import Splitter
from sklearn.model_selection import train_test_split

#拆分数据集
class RSplitter(Splitter):
    def __init__(self) -> None:
        super().__init__()

    def Split(self,data,ratio):
        
        # 获取特征和标签
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
       
        #划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)

        return X_train, X_test, y_train, y_test
    
    def run(self,data,splitter,ratio,model,performances):
        data=data.load()
        output_string=[]
        X_train, X_test, y_train, y_test = splitter.Split(data,ratio)
    
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        image_data=self.praplt(y_test,y_pred)

        for i in range(len(performances)):
            output_string.append((performances[i].name+" = {:.2f}".format(performances[i].compute(y_pred, y_test))))
        
        #print(output_string)
        return output_string,image_data


