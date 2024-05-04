import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from date import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
batch_size = 64
learning_rate = 0.01
momentum = 0.5
epochs = 200

# 预处理流程
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  
])

# 获取MNIST数据集
train_dataset = train_dataset()
test_dataset = test_dataset()

# 载入MNIST数据集
train_loder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loder = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 50),
            nn.Linear(50, 10),
        )
    
    def forward(self,x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# 实例化模型
net = Net().to(device)
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum) # 随机梯度下降优化器

def train(epoch):
    loss_num = 0.0
    total_num = 0
    acc_num = 0
    batch_idx = 0
    train_tqdm = tqdm(total=len(train_loder), desc='TRAIN: ', leave=False)
    for data in train_loder:
        inputs, target = data
        optimizer.zero_grad()

        output = net(inputs.to(device))
        loss = criterion(output, target.to(device))

        loss.backward()
        optimizer.step()

        loss_num += loss.item()
        _, predicted = torch.max(output.data, dim=1)
        total_num += inputs.shape[0]
        acc_num += (predicted == target.to(device)).sum().item()

        if batch_idx % 4 == 3:
            train_tqdm.set_postfix(Loss=f'{loss_num / 300:.4f}', acc=f'{100 * acc_num / total_num:.4f}%')
            loss_num = 0.0
            total_num = 0
            acc_num = 0
        batch_idx += 1
        train_tqdm.update(1)
        torch.save(optimizer.state_dict(), './model/optimizer_Minist.pth')
    train_tqdm.close()

def test(epoch):
    acc_num = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loder, desc='TEST: ', leave=False):
            images, labels = data
            output = net(images.to(device))
            _, predicted = torch.max(output.data, dim=1)
            total += labels.size(0)
            acc_num += (predicted == labels.to(device)).sum().item()
        
        acc = acc_num / total
        tqdm.write(f'TEST: [{epoch + 1} / {epochs}]: Acc: {100 * acc:.2f}%')
        return acc

if __name__ == '__main__':
    acc_list = []
    for epoch in tqdm(range(epochs), desc='EPOCH: '):
        train(epoch)
        acc_test = test(epoch)
        acc_list.append(acc_test)
    
    # torch.save(net, './model/model_Minist.pth')
    torch.jit.save(torch.jit.script(net), './model/model_Minist.pth')
    plt.plot(acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    # import torchsummary
    # net = Net()
    # net.to(device)
    # torchsummary.summary(net, input_size=(1, 28, 28), batch_size=128, device= 'cuda')
