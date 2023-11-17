#!/usr/bin/env python
# coding: utf-8

import argparse
import importlib
import torch
import os
from v_diffusion import make_beta_schedule
from train_utils import make_visualization
import cv2

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, required=True)
    parser.add_argument("--out_file", help="Path to image.", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=False)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
    parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode (recommended non-zero values only for not distilled model).", type=float, default=0)
    return parser

def sample_images(args, make_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = make_model().to(device)

    ckpt = torch.load(args.checkpoint)
    teacher.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"] // args.time_scale
    time_scale = ckpt["time_scale"] * args.time_scale
    del ckpt
    print("Model loaded.")

    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timesteps).to(device)
    M = importlib.import_module("v_diffusion")
    D = getattr(M, args.diffusion)
    sampler = "ddpm" if not args.clipped_sampling else "clipped"
    teacher_diffusion = D(teacher, betas, time_scale=time_scale, sampler=sampler)

    image_size = list(teacher.image_size)
    image_size[0] = args.batch_size

    img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clipping_value)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(args.out_file, img)

    print("Finished.")

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    sample_images(args, make_model)
