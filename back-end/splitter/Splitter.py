import base64
from io import BytesIO
import json
from matplotlib import pyplot as plt
class Splitter:
    def __init__(self) -> None:
        pass
    def praplt(self, y_test, y_pred):
        index = range(len(y_test))
        plt.scatter(index, y_test, label='True Values')
        plt.scatter(index, y_pred, color='r', label='Predicted Values')
        plt.xlabel('index')
        plt.ylabel('y')
        plt.title('Comparision of true values and predict values')
        plt.legend()
        #plt.show()

        #plt.savefig('output.png')
        # with open('output.png', 'rb') as f:
        #     image_data = f.read()
        # encoded_image = base64.b64encode(image_data).decode('utf-8')

        image_stream = BytesIO()
        plt.savefig(image_stream, format='png', dpi=80)
        encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
        #print(encoded_image)
        plt.close()  # 关闭图像，释放资源

        """  # 解码图片数据
        image_data = base64.b64decode(encoded_image)  # 这里替换为你的图片编码
        print(image_data)
        # 保存为图片文件
        with open("image.png", "wb") as f:
            f.write(image_data)

        # 显示图片
        image = Image.open("image.png")
        image.show() """

        data = {
            'image': encoded_image,
            'other_data': 'other_value'
        }
        json_data = json.dumps(data)
        
        return json_data

            




