from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from utils import set_all_seeds, set_deterministic
from dataloader import get_dataloaders_celeba
from dcgan import DCGAN


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

RANDOM_SEED = 42
GENERATOR_LEARNING_RATE = 0.0002
DISCRIMINATOR_LEARNING_RATE = 0.0002

NUM_EPOCHS = 10
BATCH_SIZE = 128

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 64, 3


set_deterministic()
set_all_seeds(RANDOM_SEED)


custom_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((160, 160)),
    torchvision.transforms.Resize([IMAGE_HEIGHT, IMAGE_WIDTH]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader, val_loader, test_loader = get_dataloaders_celeba(
    dataroot=Path("/home/alex/projects/generative_models/data/celeba"),
    batch_size=BATCH_SIZE,
    num_workers=10,
    train_transforms=custom_transforms,
    test_transforms=custom_transforms)

model = DCGAN()
model.to(DEVICE)

optim_gen = torch.optim.Adam(model.generator.parameters(),
                             betas=(0.5, 0.999),
                             lr=GENERATOR_LEARNING_RATE)

optim_discr = torch.optim.Adam(model.discriminator.parameters(),
                               betas=(0.5, 0.999),
                               lr=DISCRIMINATOR_LEARNING_RATE)


log_dict = {'train_generator_loss_per_batch': [],
            'train_discriminator_loss_per_batch': [],
            'train_discriminator_real_acc_per_batch': [],
            'train_discriminator_fake_acc_per_batch': [],
            'images_from_noise_per_epoch': []}

loss_fn = F.binary_cross_entropy_with_logits

latent_dim = 100
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=DEVICE)
logging_interval = 100

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, features in enumerate(train_loader):
        batch_size = features.size(0)

        real_images = features.to(DEVICE)
        real_labels = torch.ones(batch_size, device=DEVICE)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=DEVICE)
        fake_images = model.generator_forward(noise)
        fake_labels = torch.zeros(batch_size, device=DEVICE)
        flipped_fake_labels = real_labels

        optim_discr.zero_grad()

        discr_pred_real = model.discriminator_forward(real_images).view(-1)
        real_loss = loss_fn(discr_pred_real, real_labels)

        discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
        fake_loss = loss_fn(discr_pred_fake, fake_labels)

        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()
        optim_discr.step()


        optim_gen.zero_grad()

        discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
        gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
        gener_loss.backward()

        optim_gen.step()


        if not batch_idx % logging_interval:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f'
                       % (epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), gener_loss.item(), discr_loss.item()))

torch.save(model.state_dict(), "gan_celeba_01.pt")