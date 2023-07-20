from sklearn import svm
from sklearn import svm
from sklearn.preprocessing import StandardScaler

class SVC:
    parameter={
        'kernel 核函数类型':'',
    }
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.kernel = self.parameter['kernel 核函数类型']
        self.model = None
        self.scaler = None
    
    @classmethod
    def set_parameter(cls,dic):
        cls.parameter=dic


    def train(self, X_train, y_train):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.kernel == 'linear':
            self.model = svm.LinearSVC()
        else:
            self.model = svm.SVC(kernel=self.kernel)
        
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# class SVC:
#     def __init__(self, learning_rate=0.1):
#         self.learning_rate = learning_rate
#         self.model = None
    
#     def train(self, X_train, y_train):
#         self.model = svm.SVC(kernel='linear')
#         self.model.fit(X_train, y_train)
    
#     def predict(self, X):
#         return self.model.predict(X)



