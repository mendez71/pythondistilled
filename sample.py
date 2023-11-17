import argparse
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
import torch
import os
import cv2
from copy import deepcopy

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--output_folder", help="Folder to save images.", type=str, default="./images/")
    parser.add_argument("--num_images", help="Number of images to generate.", type=int, default=4)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=False)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
    parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode.", type=float, default=0)
    return parser

def sample_images(args, make_model):
    device = torch.device("cuda")
    teacher = make_model().to(device)

    # Load the model checkpoint
    ckpt = torch.load(args.checkpoint)
    teacher.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"] // args.time_scale
    time_scale = ckpt["time_scale"] * args.time_scale
    del ckpt
    print("Model loaded.")

    # Create diffusion model
    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timesteps).to(device)
    M = importlib.import_module("v_diffusion")
    D = getattr(M, args.diffusion)
    sampler = "ddpm" if not args.clipped_sampling else "clipped"
    teacher_diffusion = D(teacher, betas, time_scale=time_scale, sampler=sampler)

    # Generate and save images
    for i in range(args.num_images):
        image_size = deepcopy(teacher.image_size)
        image_size[0] = args.batch_size
        img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clipping_value)
        if img.shape[2] == 1:
            img = img[:, :, 0]
        output_file = os.path.join(args.output_folder, f"image_{i}.png")
        cv2.imwrite(output_file, img)
        print(f"Image {i} saved to {output_file}")

    print("Finished.")

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    sample_images(args, make_model)
