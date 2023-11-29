import torch
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x, node_rep=None, **kwargs):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res, node_rep=node_rep, **kwargs)
        x = x + x_res

        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x, node_rep=None):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x, node_rep=None):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        x = ops.u_mul_e_sum(graph, x, norm_coefs)

        x = self.feed_forward_module(graph, x)

        return x


class BDLModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, dropout, time=1, **kwargs):
        super().__init__()
        self.bundle_dim = bundle_dim
        self.dim = dim
        self.time = time
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x, node_rep):
        num_nodes = x.shape[0]
        num_bundles = node_rep.shape[1]
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        vector_field = x.reshape(num_nodes, num_bundles, self.bundle_dim, -1)  # works since self.dim divisible by bundle dim

        ## Option 1:
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)

        h = vector_field.reshape(num_nodes, self.dim)
        for _ in range(self.time):
            h = ops.u_mul_e_sum(graph, h, norm_coefs)
        vector_field = h.reshape(num_nodes, num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3), vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.dim)

        # ### Option 2:
        # vector_field = vector_field.sum(dim=0)
        # vector_field = torch.einsum('abcd, bde -> abce', node_rep.transpose(2, 3), vector_field)  # inverse is transpose
        # h = vector_field.reshape(num_nodes, self.dim)

        x = self.feed_forward_module(graph, h)
        return x



class SAGEModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x, node_rep=None):
        message = ops.copy_u_mean(graph, x)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x

class BDLSAGEModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, bundle_dim, dropout, time=1, num_att=5, **kwargs):
        super().__init__()
        self.bundle_dim = bundle_dim
        self.dim = dim
        self.time = time
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

        self.num_att = num_att
        self.attention = torch.nn.Parameter(torch.rand([self.num_att]))
        self.register_parameter("attention", self.attention)
        assert self.time > 20

    def forward(self, graph, x, node_rep):
        num_nodes = x.shape[0]
        num_bundles = node_rep.shape[1]
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        vector_field = x.reshape(num_nodes, num_bundles, self.bundle_dim,
                                 -1)  # works since self.dim divisible by bundle dim

        ## Option 1:
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)

        h = vector_field.reshape(num_nodes, self.dim)
        different_times = torch.empty([self.num_att, num_nodes, self.dim], device=x.device, dtype=x.dtype)
        for t in range(1, self.time+1):
            h = ops.u_mul_e_sum(graph, h, norm_coefs)
            if t == 1:
                different_times[0] = h
            elif t == 2:
                different_times[1] = h
            elif t == 5:
                different_times[2] = h
            elif t == 20:
                different_times[3] = h
            elif t == self.time+1:
                different_times[4] = h

        attention = self.attention.softmax(dim=0)
        h = torch.einsum('a, acb -> cb', attention, different_times)

        # Importantly, the feedforward is before pulling back
        h = self.feed_forward_module(graph, h)

        # Then we pull back
        vector_field = h.reshape(num_nodes, num_bundles, self.bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        message = vector_field.reshape(num_nodes, self.dim)

        x = torch.cat([x, message], axis=1)
        return x

def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')


class GATModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x, node_rep=None):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.feed_forward_module(graph, x)

        return x


class GATSepModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x, node_rep=None):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x, node_rep=None):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, values, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.output_linear(x)
        x = self.dropout(x)

        return x


class TransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x, node_rep=None):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        message = ops.u_mul_e_sum(graph, values, attn_probs)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.output_linear(x)
        x = self.dropout(x)

        return x
