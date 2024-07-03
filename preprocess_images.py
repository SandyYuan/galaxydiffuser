import numpy as np
from tqdm import tqdm
from astropy.io import fits
from glob import glob
import re
from os.path import basename, getsize
from matplotlib.image import imread

def check_for_corruption(ar):
    '''
    Check if the galaxies are there and then remove the image if its
    not there.
    
    If more than 1/3 of the pixels are duds we don't want that gal.
    '''
    nanned = np.any(~np.isfinite(ar))
    zeroed = np.sum(ar == 0) > (ar.size*0.3)
    return np.any((nanned, zeroed))

for fi in tqdm(sorted(glob('/pscratch/sd/v/virajvm/sandy_imgs/grz_cutouts_jpg/grz_dr9_id_*.jpg'))):
    if getsize(fi) < 5000:
        print(f'{fi} is empty, not dealing with that...')
        continue
    try:
        a = imread(fi)
        g = a[:,:,0]
        r = a[:,:,1]
        z = a[:,:,2]

        if any(map(check_for_corruption, (g, r, z))):
            print(f'Removing corrupted/missing pixels image: {fi}')
            continue

    except Exception as e:
        print(f'Problem with loading, skipping that one...\n{e}')
        continue

    gal = np.stack([g, r, z], axis=0)
    gal = gal[:, gal.shape[1]//2 - 64:gal.shape[1]//2 + 64, gal.shape[2]//2 - 64:gal.shape[2]//2 + 64]

    np.save(f'/pscratch/sd/s/sihany/desiimages/{basename(fi)[11:-5]}.npy', gal)