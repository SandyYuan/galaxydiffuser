import argparse
from ddpm.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Galaxies
import torch
import wandb

wandb.init()

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(
        dim = 128,
        dim_mults = (1, 1, 2, 2, 2, 4, 4, 4)
    ).to(device=DEVICE)

    diffusion = GaussianDiffusion(
        model,
        image_size =128,
        timesteps = 1000,
        loss_type = 'l2'
    ).to(device=DEVICE)


    trainer = Trainer(
        diffusion,
        '/pscratch/sd/s/sihany/desiimages/',
        logdir = '/pscratch/sd/s/sihany/logs/desi/',
        image_size = 128,
        train_batch_size = 16,
        train_lr = 1e-4,
        train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        step_start_ema = 5000,
        save_every = 5000,
        sample_every = 3000,
        num_workers=16,
        # rank = [0, 1, 2]
    )

    trainer.load(38000)

    trainer.train()