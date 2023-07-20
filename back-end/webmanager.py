import asyncio
import websockets
from logger import Logger 
import json

class Webmanager:
    def __init__(self) -> None:
        self.logger=Logger()

    def connect(self,factory):
        #构建websocket服务器
        async def echo(websocket, path):
            #对于接收到的信息，传回
            async for message in websocket:
                #print("I got your message: {}".format(message))

                if message=="0":
                    #需要返回分类字典keys
                    factory.set_type(0)
                    await websocket.send(json.dumps(factory.c_inspect()))
                    
                elif message=="1":
                    #需要返回回归字典keys
                    factory.set_type(1)
                    await websocket.send(json.dumps(factory.r_inspect()))
                    pass
                else:
                    #返回的是配置文件
                    message=json.loads(message)
                    #print(message)
                    factory.set_configuration(message)
                    factory.Run(self.logger)
                    #print(logger.output())
                    await websocket.send(json.dumps(self.logger.output()))
                #print(message)

        asyncio.get_event_loop().run_until_complete(websockets.serve(echo, 'localhost', 8080))
        asyncio.get_event_loop().run_forever()
