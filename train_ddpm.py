from datasets import load_dataset, concatenate_datasets
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
    return {"image": images, "dr8_id": examples['dr8_id'], "spec_z": examples['spec_z'], 
            "mass_inf_photoz": examples['mass_inf_photoz'], "sfr_inf_photoz": examples['sfr_inf_photoz']}

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # meta data
    metadata = load_dataset("parquet", data_files="/pscratch/sd/s/sihany/smithgalaxies/galaxies/metadata.parquet", split="train")
    meta_sorted = metadata.sort("dr8_id")
    # meta_sorted = meta_sorted.with_format("torch", device=DEVICE)


    ds = load_dataset("parquet", data_files={'train': '/pscratch/sd/s/sihany/smithgalaxies/galaxies/data/train*'}, split="train[:12%]")

    ds = ds.sort("dr8_id")
    # ds = ds.with_format("torch", device=DEVICE)
    print(len(ds))
    ids_train = set(ds['dr8_id'])
    meta_train = meta_sorted.filter(lambda example: example["dr8_id"] in ids_train, num_proc = 4)
    assert len(meta_train['dr8_id']) == len(ds['dr8_id'])
    assert meta_train['dr8_id'] == ds['dr8_id']

    # select only rows with good specz
    inds = [example is not None for example in meta_train['spec_z']]
    import numpy as np
    inds = np.arange(len(meta_train))[np.array(inds)]

    ds = ds.select(inds)
    meta_train = meta_train.select(inds)

    # merge datasets
    ds = concatenate_datasets([ds, meta_train.select_columns(['spec_z', 'mass_inf_photoz', 'sfr_inf_photoz'])], axis = 1)
    # further filtering
    ds = ds.filter(lambda example: (example["mass_inf_photoz"] > 0) & (example["sfr_inf_photoz"] > -10), num_proc = 4)
    
    # send to cuda
    ds = ds.with_format("torch", device=DEVICE)

    # Preprocessing the datasets and DataLoaders creation.
    ds.set_transform(transform_images)

    dl = data.DataLoader(ds, batch_size = 16, shuffle=True)
    
    wandb.init()
    
    model = Unet(
        dim = 128,
        dim_mults = (1, 1, 2, 2, 2, 4, 4, 4),
        dim_cond = 3
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
        logdir = '/pscratch/sd/s/sihany/logs/desi_cond1/',
        image_size = 128,
        train_batch_size = 16,
        train_lr = 1e-4,
        train_num_steps = 200000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        step_start_ema = 5000,
        save_every = 5000,
        sample_every = 6000,
        num_workers=16,
        cond = True,
    )

    trainer.load(95000)

    trainer.train()