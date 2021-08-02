import argparse
import os
import random

import torch.nn as nn
import torch.cuda.amp as amp
import torch.utils.data
from tqdm import tqdm

from dataset import SRDataset
from loss import PerceptionLoss
from models import Generator, Discriminator
from utils import init_torch_seeds, convert_image

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# параметры датасета
augments = True
crop_size = 256
lr_img_type = 'imagenet-norm'
hr_img_type = 'imagenet-norm'
train_data_name = 'train_images_DF+FFHQ.json'

# параметры обучения модели
save_every = 20
print_every = 500
start_epoch = 0
iters = 2e5
batch_size = 16
lr = 2e-4
beta = 1e-3 
manualSeed = None
workers = 4

# параметры архитектуры модели
srresnet_checkpoint = './weights/SRResNet_16blocks_4x.pth'
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

# создаем генератор и дискриминатор
generator = Generator(n_blocks=n_blocks, scaling_factor=upscale_factor).to(device)
generator.load_state_dict(torch.load(srresnet_checkpoint)) # инициализируем генератор весами SRResNet
discriminator = Discriminator().to(device)

# инициализируем loss-ы
perception_criterion = PerceptionLoss().to(device) # MSE в пространстве фичей vgg
adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

# переводим в режим обучения
generator.train()
discriminator.train()

epochs = int(iters // len(dataloader))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))

scaler_d = amp.GradScaler()
scaler_g = amp.GradScaler()
    
for epoch in range(start_epoch, epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    g_avg_loss = 0.0
    d_avg_loss = 0.0
    for i, (lr_imgs, hr_imgs) in progress_bar:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        # сначала обучаем генератор
        optimizer_g.zero_grad()
        
        with amp.autocast():
            # получаем fake highres изображения
            sr_imgs = generator(lr_imgs)
            # в vgg19 на вход нужно подавать отнормированные изображения
            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm')
            
            fake_labels = discriminator(sr_imgs)

            # считаем loss-ы
            perception_loss = perception_criterion(sr_imgs, hr_imgs)
            adversarial_loss = adversarial_criterion(fake_labels, torch.ones_like(fake_labels))
            perceptual_loss = perception_loss + beta * adversarial_loss

        # back propagation
        scaler_g.scale(perceptual_loss).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        
        # обучаем дискриминатор
        optimizer_d.zero_grad()
        
        with amp.autocast():
            hr_labels = discriminator(hr_imgs)
            fake_labels = discriminator(sr_imgs.detach())
            
            # Binary Cross-Entropy loss
            adversarial_loss = adversarial_criterion(fake_labels, torch.zeros_like(fake_labels)) + \
                               adversarial_criterion(hr_labels, torch.ones_like(hr_labels))
        
        scaler_d.scale(adversarial_loss).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()

        d_avg_loss += adversarial_loss.item()
        g_avg_loss += perceptual_loss.item()

        progress_bar.set_description(f"[{epoch + 1}/{epochs}][{i + 1}/{len(dataloader)}] "
                                     f"Loss_D: {adversarial_loss.item():.4f} Loss_G: {perceptual_loss.item():.4f} ")

        total_iter = len(dataloader) * epoch + i
        
        if i % print_every == 0 and i != 0:
            print(f"Avg Loss_G: {(g_avg_loss/(i+1)):.4f} Avg Loss_D: {(d_avg_loss/(i+1)):.4f}")

            
    # сохраняем модели
    if (epoch+1)%save_every == 0:
        torch.save(generator.state_dict(), 
                   f"./weights/SRGAN_{n_blocks}blocks_{upscale_factor}x_epoch{(epoch+1)}.pth")
    else:
        torch.save(generator.state_dict(), f"./weights/SRGAN_{n_blocks}blocks_{upscale_factor}x.pth")
