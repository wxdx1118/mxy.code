class Logger():
    def __init__(self) -> None:
        self.res={
            'id':2,
            'res':[]
        }

    def set_output(self,result):
        self.res['res']=result

    def output(self):
        return self.res
