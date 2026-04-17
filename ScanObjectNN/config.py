from types import SimpleNamespace
from torch import nn
import torch
from pathlib import Path

# ScanObjectNN dataset path
data_path = Path("data/h5_files")

epoch = 250
warmup = 20
batch_size = 32
learning_rate = 3e-3
label_smoothing = 0.2

lanet_args = SimpleNamespace()
lanet_args.depths = [4, 4, 4]
lanet_args.ns = [1024, 256, 64]
lanet_args.ks = [24, 24, 24]
lanet_args.dims = [96, 192, 384]
lanet_args.nbr_dims = [48, 48]  
lanet_args.bottleneck = 2048
lanet_args.num_classes = 15
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(lanet_args.depths)).split(lanet_args.depths)
lanet_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
lanet_args.bn_momentum = 0.1
lanet_args.act = nn.GELU
lanet_args.mlp_ratio = 2
lanet_args.cor_std = [2.2, 4.4, 8.8]

