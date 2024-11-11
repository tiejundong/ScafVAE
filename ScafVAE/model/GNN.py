import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add, scatter_softmax
from einops import rearrange, repeat
import torch_geometric
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import to_dense_batch
import numpy as np

from openfold.utils.precision_utils import is_fp16_enabled
from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock
from esm.esmfold.v1.misc import PairToSequence

from ScafVAE.utils.common import *
from ScafVAE.utils.data_utils import *
MAX_VALID_fp16 = 5e+4


class GateResidue(torch.nn.Module):
    def __init__(self, hidden, gate=False):
        super(GateResidue, self).__init__()
        self.gate = gate
        if gate:
            self.gate = torch.nn.Linear(hidden * 3, 1)

    def forward(self, x, res):
        if self.gate:
            g = self.gate(torch.cat((x, res, x - res), dim=-1)).sigmoid()
            return x * g + res * (1 - g)
        else:
            return x + res


class FeedForward(torch.nn.Module):
    def __init__(self, hidden, dropout=0.1, multi=1, act='leakyrelu'):
        super(FeedForward, self).__init__()
        self.FF_1 = torch.nn.Linear(hidden, hidden * multi)
        self.act = load_act(act)
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.FF_2 = torch.nn.Linear(hidden * multi, hidden)

    def forward(self, x):
        x = self.FF_1(x)
        x = self.act(x)
        x = self.dropout_layer(x)
        x = self.FF_2(x)
        x = self.act(x)
        return x


class GateNormFeedForward(torch.nn.Module):
    def __init__(self, hidden, dropout=0.1, gate=False, act='leakyrelu'):
        super(GateNormFeedForward, self).__init__()
        self.FF = FeedForward(hidden, dropout=dropout, act=act)
        self.gate = GateResidue(hidden, gate=gate)
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, x):
        x_shortcut = x
        x = self.norm(x)
        x = self.FF(x)
        x = self.gate(x, x_shortcut)
        return x


class GAT(torch.nn.Module):
    def __init__(self,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout=0.1,
                 mix_type='add',):
        super(GAT, self).__init__()
        self.n_head = n_head
        self.head_hidden = head_hidden
        self.sqrt_head_hidden = np.sqrt(self.head_hidden)

        assert mix_type in ['add', 'dot']
        self.mix_type = mix_type
        if mix_type == 'add':
            bias = False
        elif mix_type == 'dot':
            bias = True

        self.lin_qk = torch.nn.Linear(node_hidden, n_head * head_hidden * 2, bias=bias)
        self.lin_edge = torch.nn.Linear(edge_hidden, n_head * head_hidden, bias=bias)

        self.lin_node_out = torch.nn.Linear(n_head * head_hidden, node_hidden)
        self.lin_v = torch.nn.Linear(node_hidden, n_head * head_hidden)
        self.lin_edge_out = torch.nn.Linear(n_head * head_hidden, edge_hidden)

        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, edge_attr, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device)

        q, k = self.lin_qk(x).chunk(2, dim=-1)
        e = self.lin_edge(edge_attr)
        e = rearrange(e, '... i j (h d) -> ... i j h d', h=self.n_head)

        q = rearrange(q, '... i (h d) -> ... i () h d', h=self.n_head)
        k = rearrange(k, '... j (h d) -> ... () j h d', h=self.n_head)
        if self.mix_type == 'add':
            raw_att = q * k + e
        elif self.mix_type == 'dot':
            raw_att = q * k * e

        att = raw_att.sum(dim=-1) / self.sqrt_head_hidden
        att.masked_fill_(~rearrange(mask.bool(), '... j -> ... () j ()'), -get_max_tensor(att))
        att = att.softmax(dim=-2)
        att = self.dropout_layer(att)
        v = self.lin_v(x)
        v = rearrange(v, '... j (h d) -> ... j h d', h=self.n_head)
        node_out = torch.einsum('... i j h, ... j h d -> ... i h d', att, v)
        node_out = self.lin_node_out(rearrange(node_out, '... i h d -> ... i (h d)'))
        node_out = self.dropout_layer(node_out)

        edge_out = self.lin_edge_out(rearrange(raw_att, '... i j h d -> ... i j (h d)'))
        edge_out = self.dropout_layer(edge_out)

        return node_out, edge_out


class GraphTransformerBlock(torch.nn.Module):
    def __init__(self,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout=0.1,
                 gate=False,
                 mix_type='add',
                 ):
        super(GraphTransformerBlock, self).__init__()
        self.att_layer = GAT(node_hidden,
                             edge_hidden,
                             n_head,
                             head_hidden,
                             dropout=dropout,
                             mix_type=mix_type)
        self.norm_node = torch.nn.LayerNorm(node_hidden)
        self.norm_edge = torch.nn.LayerNorm(edge_hidden)
        self.gate_node = GateResidue(node_hidden, gate=gate)
        self.gate_edge = GateResidue(edge_hidden, gate=gate)

        self.ff_x = GateNormFeedForward(node_hidden, dropout=dropout, gate=gate)
        self.ff_edge = GateNormFeedForward(edge_hidden, dropout=dropout, gate=gate)

    def forward(self, x, edge_attr, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device)

        x_shortcut = x
        edge_attr_shortcut = edge_attr

        # prenorm
        x = self.norm_node(x)
        edge_attr = self.norm_edge(edge_attr)

        # att
        x, edge_attr = self.att_layer(x, edge_attr, mask)

        # gate
        x = self.gate_node(x, x_shortcut)
        edge_attr = self.gate_edge(edge_attr, edge_attr_shortcut)

        # ff
        x = self.ff_x(x)
        edge_attr = self.ff_edge(edge_attr)

        return x, edge_attr


class RGCN(torch.nn.Module):
    def __init__(self, node_hidden, edge_hidden, dropout=0.1):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(edge_hidden, node_hidden, node_hidden))
        self.b = torch.nn.Parameter(torch.randn(edge_hidden, 1, node_hidden))
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.b)

        self.norm = torch.nn.LayerNorm(node_hidden)

        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, adj, mask=None, chunk_size=None):
        if mask is not None:
            adj = adj * rearrange(mask, 'b n -> b () n ()')

        x_shortcut = x

        x = self.norm(x)
        x = torch.einsum('b i j r, b j d -> b r i d', (adj, x))
        x = torch.einsum('b r i d, r d h -> b r i h', (x, self.W))
        x = x + self.b
        x = self.dropout_layer(x)
        x = x.sum(dim=-3)
        x = F.leaky_relu(x)

        x = x + x_shortcut

        return x, adj


class GNNBlock(torch.nn.Module):
    def __init__(
            self,
            block_type,
            node_hidden,
            edge_hidden,
            n_head,
            dropout=0.1,
            mix_type='add',
    ):
        super().__init__()
        assert block_type in ['GT', 'tri']
        self.block_type = block_type

        if block_type == 'GT':
            self.block = GraphTransformerBlock(
                node_hidden=node_hidden,
                edge_hidden=edge_hidden,
                n_head=n_head,
                head_hidden=node_hidden // n_head,
                dropout=dropout,
                mix_type=mix_type,
            )
        elif block_type == 'tri':
            self.block = TriangularSelfAttentionBlock(
                sequence_state_dim=node_hidden,
                pairwise_state_dim=edge_hidden,
                sequence_head_width=node_hidden // n_head,
                pairwise_head_width=edge_hidden // n_head,
                dropout=dropout,
            )

    def forward(self, x, edge, mask=None, chunk_size=None):
        if self.block_type == 'GT':
            return self.block(x, edge, mask=mask)
        elif self.block_type == 'tri':
            return self.block(x, edge, mask=mask, chunk_size=chunk_size)


class RNNGNNBlock(torch.nn.Module):
    def __init__(
        self,
        RNN_hidden,
        GNN_block_type,  # e.g. tri-GT-GT
        node_hidden,
        edge_hidden,
        n_head,
        dropout=0.1,
        mix_type='add',
        chunk_size=None,
    ):
        super().__init__()
        self.GNN_blocks = torch.nn.ModuleList(
            [
                GNNBlock(
                    block_type=block_type,
                    node_hidden=node_hidden,
                    edge_hidden=edge_hidden,
                    n_head=n_head,
                    dropout=dropout,
                    mix_type=mix_type,
                )
                for block_type in GNN_block_type.split('-')
            ]
        )
        self.chunk_size = chunk_size

        self.GRU_cells = torch.nn.ModuleList(
            [
                torch.nn.GRUCell(
                    input_size=RNN_hidden,
                    hidden_size=RNN_hidden,
                )
                for _ in range(len(self.GNN_blocks))
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)

        pooling_hidden = (node_hidden + edge_hidden) * 1
        self.to_graph_layer = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(pooling_hidden),
                    torch.nn.Linear(pooling_hidden, RNN_hidden),
                )
                for _ in range(len(self.GNN_blocks))
            ]
        )
        self.to_x_layer = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(RNN_hidden),
                    torch.nn.Linear(RNN_hidden, node_hidden),
                )
                for _ in range(len(self.GNN_blocks))
            ]
        )
        self.to_edge_layer = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(RNN_hidden),
                    torch.nn.Linear(RNN_hidden, edge_hidden),
                )
                for _ in range(len(self.GNN_blocks))
            ]
        )

    def forward(self, x, edge, graph, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device)

        x, edge, graph = self.run(x, edge, graph, mask)

        return x, edge, graph

    def run(self, x, edge, graph, mask):
        mask_add1 = pad_one(mask)

        for i, block in enumerate(self.GNN_blocks):
            x, edge = self.to_x_edge(x, edge, graph, i)

            x, edge = block(x, edge, mask=mask_add1, chunk_size=self.chunk_size)

            x, edge = self.extract_x_edge(x, edge)
            graph = self.GRU_cells[i](
                self.to_graph(x, edge, mask, i),
                graph
            )
            graph = self.dropout(graph)

        edge = (edge + rearrange(edge, 'b i j d -> b j i d')) / 2

        return x, edge, graph

    def extract_x_edge(self, x, edge):
        return x[:, :-1], edge[:, :-1, :-1]

    def to_graph(self, x, edge, mask, i):
        pair_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-2)
        pooling = torch.concat([
            (x * mask.unsqueeze(dim=-1)).sum(dim=-2) /
            (reduce(mask, 'b n -> b ()', 'sum') + EPS),
            (edge * pair_mask.unsqueeze(dim=-1)).sum(dim=[-2, -3]) /
            (reduce(pair_mask, 'b i j -> b ()', 'sum') + EPS),

            # reduce(
            #     x.masked_fill_(mask.unsqueeze(dim=-1).bool(), -1e+5),
            #     'b n d -> b d', 'max',
            # ),
            # reduce(
            #     edge.masked_fill_(pair_mask.unsqueeze(dim=-1).bool(), -1e+5),
            #     'b i j d -> b d', 'max',
            # ),
        ], dim=-1)
        pooling = self.to_graph_layer[i](pooling)
        return pooling

    def to_x_edge(self, x, edge, graph, i):
        new_x = self.to_x_layer[i](graph)
        x = torch.concat([x, new_x.unsqueeze(dim=1)], dim=1)

        mask = F.pad(
            torch.zeros(edge.shape[:3], device=edge.device),
            (0, 1, 0, 1), 'constant', 1,
        )
        edge = F.pad(
            edge,
            (0, 0, 0, 1, 0, 1), 'constant', 0,
        )
        new_edge = repeat(
            self.to_edge_layer[i](graph),
            'b d -> b i j d', i=edge.shape[1], j=edge.shape[2],
        )
        edge = edge + new_edge * mask.unsqueeze(dim=-1)

        return x, edge





















