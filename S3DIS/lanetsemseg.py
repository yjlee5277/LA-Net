import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling


def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)


class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)

    def forward(self, x):  # x(B, N, C)
        B, N, C = x.shape
        x = self.mlp(x.view(B * N, -1)).view(B, N, -1)
        return x  # x(B, N, C)


# (B, C, N) ---> (B, C, N)
class AFF(nn.Module):
    def __init__(self, dim, ratio=1):
        super().__init__()
        self.dim = dim

        self.trans = nn.Sequential(
            nn.Conv1d(self.dim, self.dim // ratio, kernel_size=1),
            nn.BatchNorm1d(self.dim // ratio),
            nn.Conv1d(self.dim // ratio, self.dim, kernel_size=1),
            nn.BatchNorm1d(self.dim),
            nn.Sigmoid()
        )

    def forward(self, x):  # [x1, x2]  (B, C, N)
        features = x[0] + x[1]
        attention_vectors = self.trans(features)
        out_features = attention_vectors * x[0] + (1 - attention_vectors) * x[1]
        return out_features  # (B, C, N)


# (B, N, C) (B, N, k) ---> (B, N, C)
class DFIL(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.local_learning = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        )

        self.global_learning = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        )

        self.aff_layer = AFF(out_dim)

        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):  # x(B, N, C)  knn(B, N, k)
        B, N, C = x.shape

        # Local Feature
        proj_x = self.proj(x)  # (B, N, C)
        x_knn = knn_edge_maxpooling(proj_x, knn, self.training)  # (B, N, C)
        x1 = self.local_learning(x_knn.permute(0, 2, 1))  # (B, C, N)

        # Global Feature
        x2 = self.global_learning(x.view(B * N, -1)).view(B, N, -1)  # (B, N, C)
        x2 = x2.permute(0, 2, 1)  # (B, C, N)

        res = self.aff_layer([x1, x2])  # (B, C, N)

        res = self.bn(res).permute(0, 2, 1)  # (B, N, C)
        return res  # (B, N, C)


# Local Aggregation = Block = DFIL + MLP
class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()
        self.depth = depth

        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)

        self.dfils = nn.ModuleList([
            DFIL(dim, dim, bn_momentum) for _ in range(depth)
        ])

        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])

        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth

        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])

    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, knn, pts=None):  # x(B, N, C)  knn(B, N, k)
        x = x + self.drop_path(self.mlp(x), 0, pts)  # 对输入进行整体特征变换 并使用Dropout路径

        # 遍历 DFIL层 和 MLP层
        for i in range(self.depth):
            # 1.对于每一个DFIL层  使用DFIL模块对输入进行局部特征传播 并使用Dropout路径 再使用残差连接
            x = x + self.drop_path(self.dfils[i](x, knn), i, pts)

            # 2.每隔一个层 使用MLP模块对输入进行特征变换 并使用Dropout路径 再使用残差连接
            if i % 2 == 1:
                x = x + self.drop_path(self.mlps[i // 2](x), i, pts)
        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()
        self.depth = depth
        self.up_depth = len(args.depths) - 1

        self.first = first = depth == 0
        self.last = last = depth == self.up_depth

        self.k = args.ks[depth]

        self.cp = cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        dim = args.dims[depth]
        nbr_in_dim = 7 if first else 3
        nbr_hid_dim = args.nbr_dims[0] if first else args.nbr_dims[1] // 2
        nbr_out_dim = dim if first else args.nbr_dims[1]

        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in_dim, nbr_hid_dim // 2, bias=False),
            nn.BatchNorm1d(nbr_hid_dim // 2, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim // 2, nbr_hid_dim, bias=False),
            nn.BatchNorm1d(nbr_hid_dim, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False),
        )
        self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if first else 0.2)
        self.nbr_proj = nn.Identity() if first else nn.Linear(nbr_out_dim, dim, bias=False)

        if not first:
            in_dim = args.dims[depth - 1]
            self.dfil = DFIL(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act)
        self.drop = DropPath(args.head_drops[depth])
        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)

    # 聚合操作包括DFIL模块和MLP模块
    def local_aggregation(self, x, knn, pts):
        x = x.unsqueeze(0)
        x = self.blk(x, knn, pts)
        x = x.squeeze(0)
        return x

    def forward(self, x, xyz, prev_knn, indices, pts_list):  # x(N, C)
        # 1.downsampling
        # 不是第一层
        if not self.first:
            ids = indices.pop()  # 获取上一层采样点的索引ids
            xyz = xyz[ids]  # 根据索引ids获取对应的 输入坐标xyz 和 输入特征x
            x = self.skip_proj(x)[ids] + self.dfil(x.unsqueeze(0), prev_knn).squeeze(0)[ids]
        # 如果是第一层
        knn = indices.pop()  # 获取上一层采样点的索引 记为knn

        # 2.spatial encoding
        N, k = knn.shape  # 采样点的邻域索引knn
        nbr = xyz[knn] - xyz.unsqueeze(1)  # 计算邻域点的相对坐标nbr
        # 如果不是第一层 则将相对坐标nbr和对应的特征进行拼接 形成新的邻域表示nbr
        nbr = torch.cat([nbr, x[knn]], dim=-1).view(-1, 7) if self.first else nbr.view(-1, 3)
        # 下面这一过程看作是对邻域信息的编码
        if self.training and self.cp:
            nbr.requires_grad_()
        nbr_embed_func = lambda x: self.nbr_embed(x).view(N, k, -1).max(dim=1)[0]  # 对邻域表示nbr进行编码
        nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr)
        x = nbr if self.first else nbr + x

        # 3.main block
        knn = knn.unsqueeze(0)  # (N, k, 1)
        pts = pts_list.pop() if pts_list is not None else None
        # 对编码后的邻域表示x进行聚合操作
        x = checkpoint(self.local_aggregation, x, knn, pts) if self.training and self.cp else self.local_aggregation(x,
                                                                                                                     knn,
                                                                                                                     pts)

        # get subsequent feature maps
        if not self.last:  # 不是最后一层
            sub_x, sub_c = self.sub_stage(x, xyz, knn, indices, pts_list)  # 获取后续特征和损失
        else:
            sub_x = sub_c = None

        # regularization
        # 进行正则化操作 约束特征之间的关系 以提高模型的泛化能力
        if self.training:
            rel_k = torch.randint(self.k, (N, 1), device=x.device)
            rel_k = torch.gather(knn.squeeze(0), 1, rel_k).squeeze(1)
            rel_cor = (xyz[rel_k] - xyz)
            rel_cor.mul_(self.cor_std)
            # print(rel_cor.std(dim=0))
            rel_p = x[rel_k] - x
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_c = sub_c + closs if sub_c is not None else closs

        # 4.upsampling
        # 完成当前层的处理之后 进行特征的上采样
        x = self.postproj(x)
        # 不是第一层
        if not self.first:
            back_nn = indices[self.depth - 1]  # 获取上一层的反向索引back_nn
            x = x[back_nn]  # 根据反向索引back_nn 将特征上采样到当前层的维度
        # 对特征进行后处理操作
        x = self.drop(x)
        sub_x = sub_x + x if sub_x is not None else x

        return sub_x, sub_c  # 输出特征sub_x   正则化损失sub_c


# LA-Net for Semantic Segmentation
class LANetSemSeg(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum) ** 0.5  # args.bn_momentum = 0.02

        self.stage = Stage(args)

        hid_dim = args.head_dim  # 256
        out_dim = args.num_classes  # 13

        self.head = nn.Sequential(
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(hid_dim, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None):
        indices = indices[:]  # 将索引列表进行复制
        x, closs = self.stage(x, xyz, None, indices, pts_list)
        if self.training:
            return self.head(x), closs
        return self.head(x)

