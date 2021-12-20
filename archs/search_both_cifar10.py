# @Date    : 2020-10-22
# @Author  : Chen Gao

from torch import nn
from archs.search_both_cifar10_building_blocks import Cell, DisBlock, OptimizedDisBlock
import sys
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.base_latent_dim = args.latent_dim // 3
        self.l1 = nn.Linear(args.latent_dim // 3, (self.bottom_width ** 2) * args.gf_dim)
        self.l2 = nn.Linear(args.latent_dim // 3, ((self.bottom_width * 2) ** 2) * args.gf_dim)
        self.l3 = nn.Linear(args.latent_dim // 3, ((self.bottom_width * 4) ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in=1)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.initialize_alphas()

    def initialize_alphas(self):
        num_cell = 3
        num_edge_normal = 5#一共7条边，normal操作边有5条
        num_ops = 7 ##normal操作数
        num_edge_up = 2# 上采样操作边 0-1，0-2
        num_up = 3#上采样有三种操作
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(num_cell, num_edge_normal, num_ops))
        self.alphas_up = nn.Parameter(1e-3 * torch.randn(num_cell, num_edge_up, num_up))
        self._arch_parameters = [self.alphas_normal, self.alphas_up]

    def arch_parameters(self):
        return self._arch_parameters

    def set_tau(self, tau):
        self.tau = tau
#将Z分为3部分，分别输入到三个cell中
    def forward(self, z):
        #h = self.l1(z[:, :self.base_latent_dim]) \
       #     .view(-1, self.ch, self.bottom_width, self.bottom_width)
        #n1 = self.l2(z[:, self.base_latent_dim:self.base_latent_dim * 2]) \
         #   .view(-1, self.ch, self.bottom_width * 2, self.bottom_width * 2)
        
        #n2 = self.l3(z[:, self.base_latent_dim * 2:]) \
            #.view(-1, self.ch, self.bottom_width * 4, self.bottom_width * 4)
        h = self.l1(z[:, :40]).view(-1, self.ch, self.bottom_width, self.bottom_width)
        n1 = self.l2(z[:, 40:80]).view(-1, self.ch, self.bottom_width * 2, self.bottom_width * 2)
        n2 = self.l3(z[:, 80:]).view(-1, self.ch, self.bottom_width * 4, self.bottom_width * 4)


        if self.args.gumbel_softmax:
            alphas_normal_pi = F.softmax(self.alphas_normal, dim=-1)
            weights_normal = F.gumbel_softmax(alphas_normal_pi, tau=self.tau, hard=False, dim=-1)
            alphas_up_pi = F.softmax(self.alphas_up, dim=-1)
            weights_up = F.gumbel_softmax(alphas_up_pi, tau=self.tau, hard=False, dim=-1)
        else:
            weights_normal = F.softmax(self.alphas_normal, dim=-1)
            weights_up = F.softmax(self.alphas_up, dim=-1)

        h1_skip_out, h1 = self.cell1(h, weights_normal[0], weights_up[0])
        h2_skip_out, h2 = self.cell2(h1 + n1, weights_normal[1], weights_up[1], (h1_skip_out,))
        _, h3 = self.cell3(h2 + n2, weights_normal[2], weights_up[2], (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.args = args
        self.ch = args.df_dim#search:dif_dim=80
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch, activation=activation, downsample=True)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=True)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=True)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        self.layer_E = nn.Sequential(
            nn.BatchNorm2d(args.df_dim),
            nn.ReLU(),
            nn.Conv2d(args.df_dim, 1, 3, 2, 1, bias=False)
        )
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

        self.initialize_alphas()

    def initialize_alphas(self):
        num_cell = 4
        num_edge_normal = 5 #一共7条边 normal边5条
        num_ops = 7 #normal操作数
        num_edge_down = 2 #下采样操作边 2-4，3-4
        num_down = 6 #下采样6种操作
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(num_cell, num_edge_normal, num_ops))
        self.alphas_down = nn.Parameter(1e-3 * torch.randn(num_cell, num_edge_down, num_down))
        self._arch_parameters = [self.alphas_normal, self.alphas_down]

    def arch_parameters(self):
        return self._arch_parameters

    def set_tau(self, tau):# 松弛温度系数
        self.tau = tau

    def forward(self, x):
        h = x

        if self.args.gumbel_softmax:
            alphas_normal_pi = F.softmax(self.alphas_normal, dim=-1)
            weights_normal = F.gumbel_softmax(alphas_normal_pi, tau=self.tau, hard=False, dim=-1)
            alphas_down_pi = F.softmax(self.alphas_down, dim=-1)
            weights_down = F.gumbel_softmax(alphas_down_pi, tau=self.tau, hard=False, dim=-1)
        else:
            weights_normal = F.softmax(self.alphas_normal, dim=-1)
            weights_down = F.softmax(self.alphas_down, dim=-1)
        doc=open("s-2.txt","a")
        print("weights_normal",weights_normal,file=doc)
        print("weights_down",weights_down, file=doc)
        h = self.block1(h, weights_normal=weights_normal[0], weights_down=weights_down[0])
        #assert not torch.any(torch.isnan(h))
        h = self.block2(h, weights_normal=weights_normal[1], weights_down=weights_down[1])
        #assert not torch.any(torch.isnan(h))
        h = self.block3(h, weights_normal=weights_normal[2], weights_down=weights_down[2])
        #assert not torch.any(torch.isnan(h))
        h = self.block4(h, weights_normal=weights_normal[3], weights_down=weights_down[3])
        # print(4)
        # assert not torch.any(torch.isnan(h))

        output_E = self.layer_E(h)
        #print(output_E)
        #print(output_E.view(-1, 1).squeeze(1))

        h = self.activation(h) #Relu函数 输出为tensor
        h = h.sum(2).sum(2) #Global average pooling
        output = self.l5(h) #全连接层输出一维向量
        return output , output_E.view(-1, 1).squeeze(1)

