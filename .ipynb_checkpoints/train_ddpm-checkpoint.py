from datasets import load_dataset
import argparse
from ddpm.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Galaxies
import torch
from torchvision import transforms
from torch.utils import data
import wandb

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


augmentations = transforms.Compose(
    [
            transforms.CenterCrop((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
    ]
)

def transform_images(examples):
    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"image": images}

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("parquet", data_files={'train': '/pscratch/sd/s/sihany/smithgalaxies/galaxies/data/train*'}, split="train")
    ds = ds.with_format("torch", device=DEVICE)
    print(f"dataset size {len(ds)}")

    ds.set_transform(transform_images)
    dl = data.DataLoader(ds, batch_size = 16, shuffle=True)
    
    wandb.init()
    
    model = Unet(
        dim = 64,
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
        dl = dl,
        logdir = '/pscratch/sd/s/sihany/logs/desi_cond/',
        image_size = 128,
        train_lr = 1e-4,
        train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        step_start_ema = 5000,
        save_every = 10000,
        sample_every = 3000,
        num_workers=16,
        # rank = [0, 1, 2]
    )

    trainer.load(210000)

    trainer.train()