import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = torch.jit.load('./model/model_Minist.pth')
net.to(device)
optimizer = torch.load('./model/optimizer_Minist.pth')
transform = transforms.Compose([  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])

def prediction(img):
    img = transform(img)
    with torch.no_grad():
        img = img.to(device, dtype=torch.float32)
        img = img.unsqueeze(0)
        output = net(img)
        _, predicted = torch.max(output.data, dim=1)
        return predicted.item()

def main():
    image = Image.open('图片地址，要求28*28*1')
    number = prediction(image)
    print(f'神经网络预测结果: {number}')
    plt.imshow(image, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()