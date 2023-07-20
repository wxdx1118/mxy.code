from DataFactory import DataFatory
from ModelFactory import ModelFatory
from PerformanceTesterFactory import PerformanceTesterFactory
from Register import Register
from SplitterFactory import SplitterFactory

class Factory_Run(PerformanceTesterFactory,ModelFatory,SplitterFactory,DataFatory):
    def __init__(self) -> None:
        super().__init__()
        self.data_factory=DataFatory()
        self.model_factory=ModelFatory()
        self.splitter_factory=SplitterFactory()
        self.performance_factory=PerformanceTesterFactory()

        res=Register()
        #注册
        self.data_factory.inputall(0,res.cla_datasets)
        self.data_factory.inputall(1,res.reg_datasets)

        self.model_factory.inputall(0,res.cla_model)
        self.model_factory.inputall(1,res.reg_model)

        self.performance_factory.inputall(0,res.cla_performances)
        self.performance_factory.inputall(1,res.reg_performances)

        self.splitter_factory.inputall(res.splitter)
        
    def Run(self,logger):
        data_name=self.configuration[0][0]
        ratio=self.configuration[1][0]
        splitter_name=self.configuration[2][0]
        learning_rate=self.configuration[3][0]
        model_name=self.configuration[4][0]
        performance_names=self.configuration[5]
        parameter_value=self.configuration[6]
        #print(data_name,'\n',ratio,'\n',splitter_name,'\n',learning_rate,'\n',model_name,'\n',performance_names,'\n')
        #print(parameter_value)
        performances=[]
        if(self.type==0):
            data = self.data_factory.cla_getData(data_name)#数据集对象
            model = self.model_factory.cla_getData(model_name)#算法模型对象
            model.set_parameter(parameter_value)
            #print(model.parameter)
            for performance_name in performance_names:#性能度量指标对象
                performances.append(self.performance_factory.cla_getData(performance_name))
        else:
            data = self.data_factory.reg_getData(data_name)#数据集对象
            model = self.model_factory.reg_getData(model_name)#算法模型对象
            model.set_parameter(parameter_value)
            for performance_name in performance_names:#性能度量指标对象
                performances.append(self.performance_factory.reg_getData(performance_name))

        #创建数据分割器实例化对象
        splitter=self.splitter_factory.getSplitter(splitter_name)

        #实例化
        model=model(learning_rate=learning_rate)
        result=splitter.run(data,splitter,ratio,model,performances)

        logger.set_output(result)