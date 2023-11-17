#!/usr/bin/env python
# coding: utf-8

import argparse
import importlib
import os
import subprocess
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from train_utils import *

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1024)
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=50000)  # Half of the previous value
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyConstantLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=-1)
    return parser

def generate_images(module_name, checkpoint_path, output_path, batch_size=1):
    """
    Function to generate images using sample.py
    :param module_name: Name of the model module
    :param checkpoint_path: Path to the model checkpoint
    :param output_path: Path to save the generated images
    :param batch_size: Batch size for image generation
    """
    subprocess.run([
        "python", "./sample.py",
        "--module", module_name,
        "--checkpoint", checkpoint_path,
        "--out_file", output_path,
        "--batch_size", str(batch_size)
    ])

def train_model(args, make_model, make_dataset):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    device = torch.device("cuda")
    train_dataset = test_dataset = InfinityDataset(make_dataset(), args.num_iters)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    teacher_ema = make_model().to(device)

    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    scheduler = make_sheduler(args)

    teacher = make_model().to(device)
    teacher_ema = make_model().to(device)

    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        teacher.load_state_dict(ckpt["G"])
        teacher_ema.load_state_dict(ckpt["G"])
        del ckpt
        print("Continue training...")
    else:
        print("Training new model...")
    init_ema_model(teacher, teacher_ema)

    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    teacher_diffusion = make_diffusion(teacher, args.num_timesteps, 1, device)
    teacher_ema_diffusion = make_diffusion(teacher, args.num_timesteps, 1, device)

    image_size = teacher.image_size

    on_iter = make_iter_callback(teacher_ema_diffusion, device, checkpoints_dir, image_size, tensorboard, args.log_interval, args.ckpt_interval, False)
    diffusion_train = DiffusionTrain(scheduler)
    diffusion_train.train(train_loader, teacher_diffusion, teacher_ema, args.lr, device, make_extra_args=make_condition, on_iter=on_iter)

    # Generate images after training
    checkpoint_path = os.path.join(checkpoints_dir, "final_checkpoint.pt")  # Adjust this path as needed
    output_path = os.path.join(checkpoints_dir, "generated_image.png")  # Adjust this path as needed
    generate_images(args.module, checkpoint_path, output_path)

    print("Finished.")

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    train_model(args, make_model, make_dataset)
