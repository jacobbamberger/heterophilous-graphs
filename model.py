import torch
from torch import nn
from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule, BDLModule, BDLSAGEModule)
from bundles.orthogonal import Orthogonal

MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule],
    'BDL': [BDLModule],
    'BDLSAGE': [BDLSAGEModule]
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x


class BDLModel(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, bundle_dim, output_dim,
                 hidden_dim_multiplier, num_heads,
                 normalization, dropout, time, num_bundles=256):

        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.hidden_dim = hidden_dim
        self.bundle_dim = bundle_dim
        if num_bundles is None:
            num_bundles = (self.hidden_dim // self.bundle_dim),
        self.num_bundles = num_bundles

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        self.orthogonal = Orthogonal(d=bundle_dim, orthogonal_map="householder")
        self.struct_encoder = Model("SAGE", 5, # TODO: add this as hyperparam
                                    hidden_dim,
                                    hidden_dim=bundle_dim**2*self.num_bundles*hidden_dim_multiplier,
                                    output_dim=bundle_dim**2*self.num_bundles,
                                    hidden_dim_multiplier=hidden_dim_multiplier,
                                    normalization="None",
                                    dropout=0.2,
                                    num_heads=0)
        self.enc_computers = nn.ModuleList()

        for _ in range(num_layers):
            self.enc_computers.append(
                FeedForwardModule(bundle_dim**2*self.num_bundles, hidden_dim_multiplier, dropout)
            )
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        bundle_dim=bundle_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        time=time)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        num_nodes = x.shape[0]
        x = self.input_linear(x)

        # enc = self.struct_encoder(graph, x) # torch.ones([x.shape[0], 1], dtype=x.dtype, device=x.device))

        x = self.dropout(x)
        x = self.act(x)

        for k, residual_module in enumerate(self.residual_modules):
            enc = self.struct_encoder(graph, x)  # torch.ones([x.shape[0], 1], dtype=x.dtype, device=x.device))
            node_rep = self.enc_computers[k](graph, enc)

            node_rep = node_rep.reshape(num_nodes * self.num_bundles,
                                        self.bundle_dim, self.bundle_dim)
            node_rep = self.orthogonal(node_rep)
            node_rep = node_rep.reshape(num_nodes, self.num_bundles,
                                                    self.bundle_dim, self.bundle_dim)  # want it to be one matrix per channel per node

            x = residual_module(graph, x, node_rep)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x
