import argparse
import os
import random

import torch.nn as nn
import torch.cuda.amp as amp
import torch.utils.data
from tqdm import tqdm

from dataset import SRDataset
from models import Generator
from utils import init_torch_seeds

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# параметры датасета
augments = {
    'rotation': False,  #поворот на 90, 180 или 270 градусов
    'hflip' : True      #горизонтальное отражение изображения
}
crop_size = 256
lr_img_type = 'imagenet-norm'
hr_img_type = '[-1, 1]'
train_data_name = './jsons/train_images.json'

# параметры обучения модели
save_every = 20
print_every = 2000
start_epoch = 0
iters = 2e5
batch_size = 16
lr = 2e-4
manualSeed = None
workers = 4

# параметры структуры модели
upscale_factor = 4
n_blocks = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Зададим рандомный seed, чтобы была возможность воспроизвести результат
if manualSeed is None:
    manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
init_torch_seeds(manualSeed)

dataset = SRDataset(crop_size=crop_size, scaling_factor=upscale_factor,
                    lr_img_type=lr_img_type, hr_img_type=hr_img_type,
                    train_data_name=train_data_name, augments=augments)

dataloader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         num_workers=int(workers))

# создаем объект нашей нейросети
generator = Generator(n_blocks=n_blocks, scaling_factor=upscale_factor).to(device)

# создаем loss, оптимизатор и scaler
content_criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
scaler = amp.GradScaler()

# переводим в режим обучения
generator.train()


print(f"[*] Start training SRResNet model based on MSE loss.")
print("device: {}".format(device))
psnr_epochs = int(iters // len(dataloader))
for epoch in range(start_epoch, psnr_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    avg_loss = 0.0
    for i, (lr_imgs, hr_imgs) in progress_bar:
        # получаем lowres(input) и highres(target) изображения
        lr = lr_imgs.to(device, non_blocking=True)
        hr = hr_imgs.to(device, non_blocking=True)
    
        optimizer.zero_grad()
        with amp.autocast():
            # генерируем "фейковые" изображения высокого разрешения из входного изображения низкого разрешения
            sr = generator(lr)
            # считаем попиксельную MSE у фейкового и настоящего изображений высокого разрешения
            mse_loss = content_criterion(sr, hr)
            
        # считаем градиенты, используя scaler
        scaler.scale(mse_loss).backward()
        # обновляем веса модели
        scaler.step(optimizer)
        scaler.update()

        avg_loss += mse_loss.item()
        progress_bar.set_description(f"[{epoch + 1}/{psnr_epochs}][{i + 1}/{len(dataloader)}] "
                                     f"MSE loss: {mse_loss.item():.4f}")
        total_iter = len(dataloader) * epoch + i
        
        if i % print_every == 0 and i!=0:
            print(f"MSE loss: {(avg_loss/(i+1)):.4f}")
    
    # сохраняем модели
    if (epoch+1)%save_every == 0:
        torch.save(generator.state_dict(),
                   f"./weights/SRResNet_{n_blocks}blocks_{upscale_factor}x_epoch{(epoch+1)}.pth")
    else:
        torch.save(generator.state_dict(), f"./weights/SRResNet_{n_blocks}blocks_{upscale_factor}x.pth")

