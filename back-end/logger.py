class Logger():
    def __init__(self) -> None:
        self.res={
            'id':2,
            'res':[],
            'pho':[]
        }

    def set_output(self,result,photo):
        self.res['res']=result
        self.res['pho']=photo

    def output(self):
        return self.res
