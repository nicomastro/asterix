# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""



from email.policy import default
import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import glob
import sys
from pathlib import Path
#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl',  default='../codigo/artgan-project/code/models/ffhq2.pkl', help='Network pickle filename', required=False)
@click.option('--rows', 'row_seeds', type=num_range,  default='1', help='Random seeds to use for image rows', required=True)
@click.option('--cols', 'col_seeds', type=num_range,  default='6', help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='6-17', show_default=True)#0-6
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--input', type=str, required=True)
def generate_style_mix(
    network_pkl: str,
    row_seeds: List[int],
    col_seeds: List[int],
    col_styles: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    input:str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python3 style_mixing.py --outdir=testeo_basico/mix--rows=1 --cols=30  --input='testeo_basico/S105/007/latente'
    
    
    #en las columnas van la expresion
    """
    
    # hacer en OSU una funcion de encoder de style mixing quetome imagen devuelva npz  y pesos y otra funcion que dado npz y pesos nos de imagen
    
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    print('Generating W vectors...')
    all_seeds = (row_seeds + col_seeds)
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in list(set(row_seeds))])#all_seeds
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    #all_w es un tensor con [cant fotos totales, 18,512] vamos a meterle las fotos que queremos

    
    source_dir = Path(input)
    W = source_dir.glob('*.npz')
    for x in sorted(W):
        imagenes.append(npz_x)
    #osu = [W]
    #cambiamos all_w usando osu
    for n in range(all_w.shape[0]):    
        npzs.insert(n,all_w[n])
    imagenes_tensor = torch.stack(npzs,dim=0)
    #aqui los cambiamos
    all_w = imagenes_tensor    


    #sigue el style mixing normal 
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}
 #   print('diccionario',w_dict)
    print('Generating images...')
    all_images = G.synthesis(all_w, noise_mode=noise_mode)
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

    print('Saving images...')
    os.makedirs(outdir, exist_ok=True)
#    print(all_w.shape)
    for (row_seed, col_seed), image in image_dict.items():
        print(row_seed, col_seed)
        print(f'{outdir}/{row_seed}-{col_seed}.png')
        PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')

          
    
    print('Saving image grid...')
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(f'{outdir}/grid.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_style_mix() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
