# Implementation of "Progressive Distillation for Fast Sampling of Diffusion Model" with the Uniform Distilled model implemented

In order to run the experiments follow the instruction in the following Google Colab:
You can view the Colab notebook by clicking [here](https://colab.research.google.com/drive/1R01_CbCNRlVGdpoRhVjQaoZQiytBTh8H#scrollTo=q2IfWrzArbZJ).


Difference of this implementation and the official one:
DDPM model was used without authors modification.
Rosanality DDPM images sampler was used by default.


The files that have been copied from other directories are: 
celeba_dataset.py
celeba_u.py
celeba_u_script.ipynb
celeba_u_script.sh
distillate.py
moving_average.py
prepare_data.py
requirements.txt
sample.py
strategies.py
train.py
train_utils.py
unet_ddpm.py
v_diffusion.py
From: https://github.com/Hramchenko/diffusion_distiller/tree/main

Modified:
v_diffusion.py, this file has been re implemented, from performing a Gaussian Distribution Diffusion implementation to a Uniform Distribution implementation from 0.

Students Original Code:
Google Colab and v_diffusion.py are students original code.

I used extracted images from Celelba_256 dataset in order to calculate the PSNR score from: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256


