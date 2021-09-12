# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import argparse
import legacy
import json
from Interpolation.util_sup import *

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

# @click.command()
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--rows', 'row_seeds', type=num_range, help='Random seeds to use for image rows', required=True)
# @click.option('--cols', 'col_seeds', type=num_range, help='Random seeds to use for image columns', required=True)
# @click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='0-6', show_default=True)
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
# @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
# @click.option('--outdir', type=str, required=True)

def generate_style_mix(
    network_pkl: str,
    pose_p: str,
    pose_g: str,
    latent_p: str,
    latent_g: str,
    outdir: str,
    col_styles= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    truncation_psi=1,
    noise_mode='const'
    ):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl'
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    print('loading W vectors for person and garment...')
    pose_p, latent_p = loadData(pose_p, latent_p)
    pose_g, latent_g = loadData(pose_g, latent_g)
    
    w_avg = G.mapping.w_avg
    #latent_p = w_avg + (latent_p - w_avg) * truncation_psi
    #latent_g = w_avg + (latent_g - w_avg) * truncation_psi
    all_w = torch.cat((latent_p,latent_g), dim=0)
    w_dict = {seed: w for seed, w in zip(['p','g'], [latent_p, latent_g])}

    print('Generating images...')
    pose_p = G.encoder(pose_p)
    img_p = G.synthesis(latent_p, pose_p, noise_mode=noise_mode)
    pose_g = G.encoder(pose_g)
    img_g = G.synthesis(latent_g, pose_g, noise_mode=noise_mode)
    img_p = (img_p.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    img_g = (img_g.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {name: image[0] for name, image in zip(['p','g'], [img_p,img_g])}

    print('Generating style-mixed images...')
    
    for col in col_styles:
        #w = latent_g.clone()
        latent_p[:,col,:] = latent_g[:,col,:]
        image = G.synthesis(latent_p, pose_p, noise_mode=noise_mode)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_dict['col{}'.format(col)] = image[0].cpu().numpy()

    print('Saving images...')
    os.makedirs(outdir, exist_ok=True)
    for name, image in image_dict.items():
        PIL.Image.fromarray(image[:, :, :3], 'RGB').save(f'{outdir}/{name}.png')
        PIL.Image.fromarray(image[:, :, 3:], 'RGB').save(f'{outdir}/{name}_mask.png')



#----------------------------------------------------------------------------
def loadData(jsonpath, latent_path):

    device = torch.device('cuda')
    dtype = torch.cuda.FloatTensor

    keypoints = json.load(open(jsonpath,'r'))
    keypoints = keypoints['people'][0]['pose_keypoints_2d']
    keypoints = np.array(keypoints)
    keypoints = keypoints.reshape((-1, 3))
    heatmaps = cords_to_map(keypoints=keypoints)
    heatmaps = torch.FloatTensor(heatmaps).type(dtype).unsqueeze(0)

    z = np.load(latent_path, allow_pickle=True)
    w = torch.tensor(z['w'], device=device).to(torch.float32)
        
    return heatmaps, w

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='runs/00008-male_512x512-vogue-batch96-noaug-resumecustom/network-snapshot-005846.pkl')
    parser.add_argument('--pose_p', default='Interpolation/person/keypoints.json')
    parser.add_argument('--pose_g', default='Interpolation/garment/keypoints.json')
    parser.add_argument('--latent_p', default='Interpolation/person/projected_w.npz')
    parser.add_argument('--latent_g', default='Interpolation/garment/projected_w.npz')
    parser.add_argument('--outdir', default='out')
    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    generate_style_mix(args.network, args.pose_p, args.pose_g, args.latent_p, args.latent_g, args.outdir) 
    # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
