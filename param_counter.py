import argparse
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler

from model import Model, BDLModel
from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup

import wandb


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


num_layers = 5
input_dim = 301
hidden_dim = 512
bundle_dim = 2
output_dim = 18
hidden_dim_multiplier = 1
num_heads = 8
normalization = 'LayerNorm'
dropout = 0.2
time = 50
num_bundles = 256

model = BDLModel("BDLSAGE",
                 num_layers,
                 input_dim,
                 hidden_dim,
                 bundle_dim,
                 output_dim,
                 hidden_dim_multiplier,
                 num_heads,
                 normalization,
                 dropout,
                 time,
                 num_bundles)

print("BDLSAGE: ", get_n_params(model))

for model_name in ['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep']:
    model = Model(model_name,
                  num_layers,
                  input_dim,
                  hidden_dim,
                  output_dim,
                  hidden_dim_multiplier,
                  num_heads,
                  normalization,
                  dropout)
    print(model_name, get_n_params(model))
