import torch
import torch.nn as nn
from torchvision import transforms  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = torch.jit.load('./model/model_Minist.pth')
net.to(device)
optimizer = torch.load('./model/optimizer_Minist.pth')
transform = transforms.Compose([  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])

# 图片预处理: 1*28*28
def image_generate():
    num = str(random.randint(0,9))
    num = '0'
    font = ImageFont.truetype(os.path.join("./fonts", os.listdir("./fonts")[random.randint(0,269)]), 20)
    rotate = random.randint(-10,10)
    width = 28
    height = 28
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), str(num), font)  
    font_width, font_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - font_width - font.getbbox(num)[0]) / 2
    y = (height - font_height - font.getbbox(num)[1]) / 2
    draw.text((x, y), num, (255, 255, 255), font)
    img = img.convert('L')
    return img

def prediction(img):
    img = transform(img)
    with torch.no_grad():
        img = img.to(device, dtype=torch.float32)
        img = img.unsqueeze(0)
        output = net(img)
        _, predicted = torch.max(output.data, dim=1)
        return predicted.item()

def main():
    image = image_generate()
    number = prediction(image)
    print(f'神经网络预测结果: {number}')
    plt.imshow(image, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()