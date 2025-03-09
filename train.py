import torch
from dataset import VanGoghPhotoDataset
import os
from utils import save_checkpoint, load_checkpoint, seed_everything
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_VG, disc_PH, gen_PH, gen_VG, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    VG_reals = 0
    VG_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (vangogh, photo) in enumerate(loop):
        vangogh = vangogh.to(config.DEVICE)
        photo = photo.to(config.DEVICE)

        with torch.amp.autocast('cuda'):
            fake_photo = gen_PH(vangogh)
            D_VG_real = disc_VG(vangogh)
            D_VG_fake = disc_VG(fake_photo.detach())
            VG_reals += D_VG_real.mean().item()
            VG_fakes += D_VG_fake.mean().item()
            D_VG_real_loss = mse(D_VG_real, torch.ones_like(D_VG_real))
            D_VG_fake_loss = mse(D_VG_fake, torch.zeros_like(D_VG_fake))
            D_VG_loss = D_VG_real_loss + D_VG_fake_loss

            fake_vangogh = gen_VG(photo)
            D_PH_real = disc_PH(photo)
            D_PH_fake = disc_PH(fake_vangogh.detach())
            D_PH_real_loss = mse(D_PH_real, torch.ones_like(D_PH_real))
            D_PH_fake_loss = mse(D_PH_fake, torch.zeros_like(D_PH_fake))
            D_PH_loss = D_PH_real_loss + D_PH_fake_loss

            D_loss = (D_VG_loss + D_PH_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.amp.autocast('cuda'):
            D_VG_fake = disc_VG(fake_photo)
            D_PH_fake = disc_PH(fake_vangogh)
            loss_G_VG = mse(D_PH_fake, torch.ones_like(D_PH_fake))
            loss_G_PH = mse(D_VG_fake, torch.ones_like(D_VG_fake))

            cycle_vangogh = gen_VG(fake_photo)
            cycle_photo = gen_PH(fake_vangogh)
            cycle_vangogh_loss = l1(vangogh, cycle_vangogh)
            cycle_photo_loss = l1(photo, cycle_photo)

            identity_vangogh = gen_VG(vangogh)
            identity_photo = gen_PH(photo)
            identity_vangogh_loss = l1(vangogh, identity_vangogh)
            identity_photo_loss = l1(photo, identity_photo)

            G_loss = (
                loss_G_VG + loss_G_PH +
                cycle_vangogh_loss * config.LAMBDA_CYCLE +
                cycle_photo_loss * config.LAMBDA_CYCLE +
                identity_vangogh_loss * config.LAMBDA_IDENTITY +
                identity_photo_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            os.makedirs("saved_images", exist_ok=True)
            save_image(fake_photo * 0.5 + 0.5, f"saved_images/photo_{idx}.png")
            save_image(fake_vangogh * 0.5 + 0.5, f"saved_images/vangogh_{idx}.png")

        loop.set_postfix(VG_real=VG_reals / (idx + 1), VG_fake=VG_fakes / (idx + 1))

def main():
    seed_everything()
    disc_VG = Discriminator(in_channels=3).to(config.DEVICE)
    disc_PH = Discriminator(in_channels=3).to(config.DEVICE)
    gen_PH = Generator(img_channels=3, num_residuals=6, num_features=32).to(config.DEVICE)
    gen_VG = Generator(img_channels=3, num_residuals=6, num_features=32).to(config.DEVICE)

    opt_disc = optim.Adam(list(disc_VG.parameters()) + list(disc_PH.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_PH.parameters()) + list(gen_VG.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_VG, gen_VG, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_PH, gen_PH, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_VG, disc_VG, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_PH, disc_PH, opt_disc, config.LEARNING_RATE)

    dataset = VanGoghPhotoDataset(root_vangogh="dataset/vangogh2photo/trainA", root_photo="dataset/vangogh2photo/trainB", transform=config.transforms)
    val_dataset = VanGoghPhotoDataset(root_vangogh="dataset/vangogh2photo/testA", root_photo="dataset/vangogh2photo/testB", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    g_scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')
    d_scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_fn(disc_VG, disc_PH, gen_PH, gen_VG, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        if config.SAVE_MODEL:
            save_checkpoint(gen_VG, opt_gen, filename=config.CHECKPOINT_GEN_VG)
            save_checkpoint(gen_PH, opt_gen, filename=config.CHECKPOINT_GEN_PH)
            save_checkpoint(disc_VG, opt_disc, filename=config.CHECKPOINT_CRITIC_VG)
            save_checkpoint(disc_PH, opt_disc, filename=config.CHECKPOINT_CRITIC_PH)

if __name__ == "__main__":
    main()
