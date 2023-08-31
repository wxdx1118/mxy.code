import numpy as np
from splitter.Splitter import Splitter
from sklearn.model_selection import KFold

class KSplitter(Splitter):
    def __init__(self) -> None:
        super().__init__()

    def Split(self,data,n_splits):

        # 获取特征和标签
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # 初始化K折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True)

        # 分割数据集
        splits = []
        for train_index, test_index in kf.split(X):
            # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            splits.append((X_train, X_test, y_train, y_test))
            #print(X_train, X_test, y_train, y_test)

        return splits
    
    def run(self,data,splitter,n_splits,model,performances):
        data=data.load()
        n=0
        performance_sum=[]
        pfm=[]
        output_string=[]
        Mean_output=[]

        for per in performances:
            performance_sum.append(0)
            pfm.append(0)
            #print(performance_sum)
        for X_train, X_test, y_train, y_test in splitter.Split(data,n_splits):

            # 在训练集上训练模型
            model.train(X_train, y_train)

            # 在测试集上进行预测
            y_pred = model.predict(X_test)
            for i in range(len(performances)):
                #print(performances[i])
                pfm[i]=performances[i].compute(y_pred, y_test)
                performance_sum[i]=performance_sum[i]+performances[i].compute(y_pred, y_test)
                #performance_sum[i]=performance_sum[i]+performances[i].compute(y_pred, y_test)
                #print(performance_sum[i],pfm)
            
            n=n+1

            # 根据索引将特定内容和原始列表中的元素一一对应，并连接为一个字符串以逗号分隔
            output_string.append(("Fold "+str(n)+': ') + (', '.join([(performances[i].name+'=') + str(round(item,2)) for i, item in enumerate(pfm)])))
        
        image_data=self.praplt(y_test,y_pred)

        for i in range(len(performances)):
            output_string.append(("Mean "+performances[i].name+": {:.2f}".format(performance_sum[i]/n) ))

        #print(output_string)
        return output_string,image_data
      
