"""
@Author
Sai Tejeswar Reddy
https://github.com/saitejeswar1

Credits:
https://github.com/aladdinpersson

"""

import argparse
import torch
from dataset import DayNightDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
torch.cuda.empty_cache()
import os
#import gc

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='choose the mode to train or eval')
opt = parser.parse_args()
print(opt)


def test_fn( gen_N, gen_D, loader):
    D_reals = 0
    D_fakes = 0
    loop = tqdm(loader, leave=True)
    print(config.DEVICE)
    
    if not os.path.exists('output/fake_nights'):
        os.makedirs('output/fake_nights')
    if not os.path.exists('output/fake_days'):
        os.makedirs('output/fake_days')

    for idx, (night, day) in enumerate(loop):
        night = night.to(config.DEVICE)
        day = day.to(config.DEVICE)
    

        # Train Discriminators D and N
        with torch.cuda.amp.autocast():
            fake_day = gen_D(night)
            fake_night = gen_N(day)

        save_image(fake_day*0.5+0.5, f"output/fake_days/day_{idx}.png")
        save_image(fake_night*0.5+0.5, f"output/fake_nights/night_{idx}.png")


        loop.set_postfix(D_real=D_reals/(idx+1), D_fake=D_fakes/(idx+1))


    

def train_fn(disc_D, disc_N, gen_N, gen_D, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    D_reals = 0
    D_fakes = 0
    loop = tqdm(loader, leave=True)
    print(config.DEVICE)

    for idx, (night, day) in enumerate(loop):
        night = night.to(config.DEVICE)
        day = day.to(config.DEVICE)
    

        # Train Discriminators D and N
        with torch.cuda.amp.autocast():
            fake_day = gen_D(night)
            D_D_real = disc_D(day)
            D_D_fake = disc_D(fake_day.detach())
            D_reals += D_D_real.mean().item()
            D_fakes += D_D_fake.mean().item()
            D_D_real_loss = mse(D_D_real, torch.ones_like(D_D_real))
            D_D_fake_loss = mse(D_D_fake, torch.zeros_like(D_D_fake))
            D_D_loss = D_D_real_loss + D_D_fake_loss

            fake_night = gen_N(day)
            D_N_real = disc_N(night)
            D_N_fake = disc_N(fake_night.detach())
            D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
            D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
            D_N_loss = D_N_real_loss + D_N_fake_loss

            # put it togethor
            D_loss = (D_D_loss + D_N_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators D and N
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_D_fake = disc_D(fake_day)
            D_N_fake = disc_N(fake_night)
            loss_G_D = mse(D_D_fake, torch.ones_like(D_D_fake))
            loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))

            # cycle loss
            cycle_night = gen_N(fake_day)
            cycle_day = gen_D(fake_night)
            cycle_night_loss = l1(night, cycle_night)
            cycle_day_loss = l1(day, cycle_day)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_night = gen_N(night)
            identity_day = gen_D(day)
            identity_night_loss = l1(night, identity_night)
            identity_day_loss = l1(day, identity_day)

            # add all togethor
            G_loss = (
                loss_G_N
                + loss_G_D
                + cycle_night_loss * config.LAMBDA_CYCLE
                + cycle_day_loss * config.LAMBDA_CYCLE
                + identity_day_loss * config.LAMBDA_IDENTITY
                + identity_night_loss * config.LAMBDA_IDENTITY
            )
        
        #torch.cuda.empty_cache()
        #gc.collect()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            save_image(fake_day*0.5+0.5, f"saved_images/day_{idx}.png")
            save_image(fake_night*0.5+0.5, f"saved_images/night_{idx}.png")

        loop.set_postfix(D_real=D_reals/(idx+1), D_fake=D_fakes/(idx+1))



def main():
    disc_D = Discriminator(in_channels=3).to(config.DEVICE) # Disctriminator for classifying Day images
    disc_N = Discriminator(in_channels=3).to(config.DEVICE) # Disctriminator for classifying Night images
    gen_N = Generator(img_channels=3, num_residuals=9).to(config.DEVICE) # Generate Night images, takes in day image
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE) # Generate Day images, takes in night image
    opt_disc = optim.Adam(
        list(disc_D.parameters()) + list(disc_N.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_N.parameters()) + list(gen_D.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_D, gen_D, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_N, gen_N, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_D, disc_D, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_N, disc_N, opt_disc, config.LEARNING_RATE,
        )

    dataset = DayNightDataset(
        root_day=config.TRAIN_DIR+"/day", root_night=config.TRAIN_DIR+"/night", transform=config.transforms
    )
    val_dataset = DayNightDataset(
       root_day=config.VAL_DIR+"/day", root_night=config.VAL_DIR+"/night", transform=config.transforms    
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    # for float16 training
    # remove the bewlo 2 lines to run float32 training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    if opt.mode == 'train':
        for epoch in range(config.NUM_EPOCHS):
            train_fn(disc_D, disc_N, gen_N, gen_D, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
            
            if config.SAVE_MODEL:
                save_checkpoint(gen_D, opt_gen, filename=config.CHECKPOINT_GEN_D)
                save_checkpoint(gen_N, opt_gen, filename=config.CHECKPOINT_GEN_N)
                save_checkpoint(disc_D, opt_disc, filename=config.CHECKPOINT_CRITIC_D)
                save_checkpoint(disc_N, opt_disc, filename=config.CHECKPOINT_CRITIC_N)
    else:
        test_fn(gen_N, gen_D, val_loader)



if __name__ == "__main__":
    main()
