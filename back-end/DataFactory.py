from Factory import Factory

class DataFatory(Factory):
    def __init__(self) -> None:
        super().__init__()
        #属性elements包含一个字典，含键值对
        self.cla_elements=dict()
        self.reg_elements=dict()
    
    #注册 数据存入工厂
    #element通常是一个类
    def register(self,tp, name, element):
        #函数setdefault绑定键值对
        if(tp==0):
            self.cla_elements.setdefault(name, element)
            self.c_register(0,name)
        else:
            self.reg_elements.setdefault(name, element)
            self.r_register(0,name)
    #注册全部
    def inputall(self,tp,datasets):
        for name, dataset in datasets.items():
            self.register(tp,name, dataset)   

    #返回分类
    def inspect(self,tp:bool):
        try:
            if(tp==0):
                return self.cla_elements.keys()
            else:
                return self.reg_elements.keys()
        except TypeError:
            print("传入的类型有误")

    #返回数据集对象
    def cla_getData(self, name: str):
        try:
            return self.cla_elements[name]("./back-end/datas/"+ name + '.csv')
        except KeyError:
            # 如果elements中不包含名称为name的key，则会抛出KeyError异常
            print("选择的数据集不存在")
            raise ValueError(f"Dataset '{name}' is not available.")
    def reg_getData(self, name: str):
        try:
            return self.reg_elements[name]("./back-end/datas/"+ name + '.csv')
        except KeyError:
            # 如果elements中不包含名称为name的key，则会抛出KeyError异常
            print("选择的数据集不存在")
            raise ValueError(f"Dataset '{name}' is not available.")
     
    
    



