from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import os
from tqdm import tqdm

def image_generate(num, font, rotate):
    font = ImageFont.truetype(font, 20)
    width = 28
    height = 28
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), str(num), font)  
    font_width, font_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - font_width - font.getbbox(num)[0]) / 2
    y = (height - font_height - font.getbbox(num)[1]) / 2
    draw.text((x, y), num, (255, 255, 255), font)
    img = img.rotate(rotate)
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

font_file = './fonts'
font_filename_list = os.listdir(font_file)
num = list(range(10))
num = [str(i) for i in num]
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = torch.jit.load('./model/model_Minist.pth')
net.to(device)
optimizer = torch.load('./model/optimizer_Minist.pth')

total = 0
right = 0
bar1 = tqdm(total=len(list(range(-10, 11, 1))) * len(num) * len(font_filename_list))

for i in num:
    for j in font_filename_list:
        for k in range(-10, 11, 1):
            image = image_generate(i, os.path.join(font_file, j), k)
            number = prediction(image)
            if number == int(i):
                right += 1
            else:
                tqdm.write(f'{i}-->{number}')
                image.save('./filt/'+i+'.png')
            total += 1
            bar1.set_postfix(total=f'{total}', right=f'{right}')
            bar1.update(1)

bar1.close()
acc = right / total * 100
print(f'总计：{total}，正确的：{right}，准确率：{acc:.4f}%')