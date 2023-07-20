from Factory import Factory

class SplitterFactory(Factory):
    # 创建一个字典用于存储测试器类和对应的任务类型
    def __init__(self) -> None:
        super().__init__()
        self.elements=dict()

    def register(self,name,element):
        self.elements.setdefault(name,element)
        self.c_register(2,name)
        self.r_register(2,name)

    def inspect(self):
        return self.elements.keys()
    
    def getSplitter(self,name:str):
        Splitter_type=self.elements[name]
        if not Splitter_type:
            raise ValueError(f"Tester '{name}' is not registered.")
        return Splitter_type()
    
    def inputall(self,splitter):
        for name, splitter in splitter.items():
            self.register(name, splitter)   