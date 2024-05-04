from PIL import Image, ImageDraw, ImageFont
import random
import os
import torch  
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class MyDataset(Dataset):  
    def __init__(self):
        self.images = []

    def append(self, image, label):
        self.images.append((image, label))

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

font_file = './fonts'
font_filename_list = os.listdir(font_file)
num = list(range(10))
num = [str(i) for i in num]
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  
])

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
    return transform(img)

bar1 = tqdm(total=len(list(range(-10, 11, 1))) * len(num) * len(font_filename_list))

def train_dataset():
    dataset = MyDataset()
    for i in num:
        for j in font_filename_list:
            for k in range(-10, 11, 1):
                image = image_generate(i, os.path.join(font_file, j), k)
                dataset.append(image, int(i))
                bar1.update(1)

    bar1.close()
    return dataset

def test_dataset():
    dataset1 = MyDataset()
    for j in font_filename_list:
        for i in num:
            k = random.randint(-10, 10)
            image = image_generate(i, os.path.join(font_file, j), k)
            dataset1.append(image, int(i))

    return dataset1

# train_dataset = train_dataset()
# train_loder = DataLoader(train_dataset, batch_size=128, shuffle=True)
# print(train_dataset[0])
# # print(train_loder[0])
# for data in train_loder:
#     inputs, target = data
# print(data)
# print(inputs.shape, type(target), target.shape)
# print(len(train_loder))