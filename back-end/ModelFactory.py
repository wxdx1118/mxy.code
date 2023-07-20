from Factory import Factory

class ModelFatory(Factory):
    def __init__(self) -> None:
        super().__init__()
        self.cla_elements = dict()
        self.reg_elements = dict()

    # 注册 算法注册
    def register(self,tp, name, element):
        # 函数setdefault绑定键值对
        if(tp==0):
            self.cla_elements.setdefault(name, element)
            self.c_register(4,name)
            # print("modelfactory:",element.parameter)
            self.c_register_parameter(name,element.parameter)
        else:
            self.reg_elements.setdefault(name, element)
            self.r_register(4,name)
            self.r_register_parameter(name,element.parameter)
    # 全部注册
    def inputall(self,tp,models):
        for name, model in models.items():
            self.register(tp,name, model)

    # 返回分类
    def inspect(self,tp:bool):
        try:
            if(tp==0):
                return self.cla_elements.keys()
            else:
                return self.reg_elements.keys()
        except TypeError:
            print("传入的类型有误")

    # 返回算法模型对象
    def cla_getData(self, name: str):
        try:
            return self.cla_elements[name]
        except KeyError:
            # 如果elements中不包含名称为name的key，则会抛出KeyError异常
            print("选择的算法模型不存在")
            raise ValueError(f"model '{name}' is not available.")
    def reg_getData(self, name: str):
        try:
            return self.reg_elements[name]
        except KeyError:
            # 如果elements中不包含名称为name的key，则会抛出KeyError异常
            print("选择的算法模型不存在")
            raise ValueError(f"model '{name}' is not available.")

