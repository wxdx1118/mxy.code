from Factory import Factory

class PerformanceTesterFactory(Factory):
    # 创建一个字典用于存储测试器类和对应的任务类型
    def __init__(self) -> None:
        super().__init__()
        self.cla_elements = dict()
        self.reg_elements = dict()

    def register(self,tp,name,element):
         #函数setdefault绑定键值对
        if(tp==0):
            self.cla_elements.setdefault(name, element)
            self.c_register(5,name)
        else:
            self.reg_elements.setdefault(name, element)
            self.r_register(5,name)

    # 全部注册
    def inputall(self,tp,performances):
        for name, performance in performances.items():
            self.register(tp,name, performance)

     #返回分类
    def inspect(self,tp:bool):
        try:
            if(tp==0):
                return self.cla_elements.keys()
            else:
                return self.reg_elements.keys()
        except TypeError:
            print("传入的类型有误")
    
     #返回算法模型对象
    def cla_getData(self, name: str):
        try:
            return self.cla_elements[name]()
        except KeyError:
            # 如果elements中不包含名称为name的key，则会抛出KeyError异常
            print("选择的性能度量指标不存在")
            raise ValueError(f"Performance '{name}' is not available.")
    def reg_getData(self, name: str):
        try:
            return self.reg_elements[name]()
        except KeyError:
            # 如果elements中不包含名称为name的key，则会抛出KeyError异常
            print("选择的性能度量指标不存在")
            raise ValueError(f"Performance '{name}' is not available.")
