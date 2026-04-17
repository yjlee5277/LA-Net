from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# S3DIS dataset path
raw_data_path = Path("data/Stanford3dDataset_v1.2_Aligned_Version")
processed_data_path = raw_data_path.parent / "s3dis"

epoch = 100
warmup = 10
batch_size = 8
learning_rate = 6e-3
label_smoothing = 0.2

s3dis_args = SimpleNamespace()
s3dis_args.k = [24, 24, 24, 24]
s3dis_args.grid_size = [0.04, 0.08, 0.16, 0.32]

s3dis_args.max_pts = 30000

s3dis_warmup_args = deepcopy(s3dis_args)
s3dis_warmup_args.grid_size = [0.04, 3.5, 3.5, 3.5]

lanet_args = SimpleNamespace()
lanet_args.ks = s3dis_args.k
lanet_args.depths = [4, 4, 8, 4]
lanet_args.dims = [64, 128, 256, 512]
lanet_args.nbr_dims = [32, 32]
lanet_args.head_dim = 256
lanet_args.num_classes = 13
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(lanet_args.depths)).split(lanet_args.depths)
lanet_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
lanet_args.head_drops = torch.linspace(0., 0.15, len(lanet_args.depths)).tolist()
lanet_args.bn_momentum = 0.02
lanet_args.act = nn.GELU
lanet_args.mlp_ratio = 2

lanet_args.use_cp = False

lanet_args.cor_std = [1.6, 3.2, 6.4, 12.8]
