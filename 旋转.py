import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        """
        初始化 DataLoader 类，接收文件路径作为参数

        参数：
        file_path (str): 数据文件的路径
        """
        self.file_path = file_path

    def load_data(self):
        """
        加载数据的方法

        返回：
        data (DataFrame): 加载的数据
        """
        data = pd.read_csv(MNIST)
        return data
import  torch
from  torchvision import transforms
from torchvision.datesets import MNIST
import natplotlib.pyplot as plt
class Net (torch.nn.module):
    def __init__(self):
            super(Net,self),__init__()
            self.fc1  =  torch.nnlinear(28*28,64)
            self.fc2  =  torch.nnlinear(64*64)
            self.fc3  =  torch.nnlinear(64*64)
            self.fc3  =  torch.nnlinear(64*10)

            self.conv = torch.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
    def  forward(self.x):
             x = torch.nn.functional.relu(self.fcl1(x))
             x = torch.nn.functional.relu(self.fcl2(x))
             x = torch.nn.functional.relu(self.fcl3(x))
             x = torch.nn.functional.relu(self.fcl3(x))
             x = torch.nn.functional.log_softmax(self,fc4(x),dim=1)


             return x
 
def      get _date _loader(is _ train);
to_tensor=transforms.Compose([transforms.ToTensor()])
date_set=MNIST(root="date")







             
