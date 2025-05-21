import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_add
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from esm.esmfold.v1.misc import SequenceToPair

from ScafVAE.model.GNN import *
from ScafVAE.model.config import *
from ScafVAE.model.loss import *
from ScafVAE.utils.hub import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.graph2mol import *



class PPLPredictor(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        model_config = PPLPredictor_config if model_config is None else model_config

        self.l_node_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(global_config.l_label_node_hidden + 1, model_config.node_hidden),
            torch.nn.LeakyReLU(),
        )
        self.l_edge_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(global_config.l_label_edge_hidden + 1, model_config.edge_hidden),
            torch.nn.LeakyReLU(),
        )

        self.blocks = torch.nn.ModuleList(
            [
                GNNBlock(
                    block_type=model_config.block_type,
                    node_hidden=model_config.node_hidden,
                    edge_hidden=model_config.edge_hidden,
                    n_head=model_config.n_head,
                    dropout=global_config.dropout,
                    mix_type=model_config.gnn_att_mix_type,
                )
                for _ in range(model_config.n_block)
            ]
        )
        self.chunk_size = global_config.chunk_size

        self.x_to_edge = SequenceToPair(
            model_config.node_hidden, model_config.edge_hidden // 2, model_config.edge_hidden
        )
        self.l_masked_bond_pred_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.edge_hidden),
            torch.nn.Linear(model_config.edge_hidden, model_config.edge_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(model_config.edge_hidden),
            torch.nn.Linear(model_config.edge_hidden, global_config.l_label_edge_hidden),
        )

        self.loss_object = PPLLossFunction()

    def forward(self, dic_data, epoch=-1):
        dic_data = self.train_ppl(dic_data)

        grad_loss, dic_log = self.loss_object(dic_data, epoch=epoch)

        return dic_data, grad_loss, dic_log

    def train_ppl(self, dic_data):
        x_init = F.pad(dic_data['l_x_init'], (0, 1), 'constant', 0)
        # x_init = self.init_blank_x(dic_data['l_x_init'])
        edge_init = dic_data['l_edge_single_masked_init']
        mask = dic_data['l_mask']

        dic_data['l_edge_single_masked_pred'] = self.pred(x_init, edge_init, mask=mask)

        return dic_data

    def init_blank_x(self, ref_x):
        x_init = torch.ones(
            ref_x.shape[0],
            ref_x.shape[1],
            ref_x.shape[2] + 1,
            device=ref_x.device,
        )
        return x_init

    def pred(self, x, edge, mask=None):
        x = self.l_node_embed_layer(x)
        edge = self.l_edge_embed_layer(edge)

        for i, block in enumerate(self.blocks):
            x, edge = block(x, edge, mask=mask, chunk_size=self.chunk_size)

        edge = edge + self.x_to_edge(x)
        edge = (edge + rearrange(edge, '... i j d -> ... j i d')) / 2

        pred = self.l_masked_bond_pred_layer(edge)

        return pred



class ScafEncoder(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        self.l_node_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(global_config.l_label_node_hidden + 1, model_config.aa_node_hidden),
            torch.nn.LeakyReLU(),
        )
        self.l_edge_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(global_config.l_label_edge_hidden + 1, model_config.aa_edge_hidden),
            torch.nn.LeakyReLU(),
        )
        self.aa_blocks = torch.nn.ModuleList(
            [
                GNNBlock(
                    block_type=block_type,
                    node_hidden=model_config.aa_node_hidden,
                    edge_hidden=model_config.aa_edge_hidden,
                    n_head=model_config.n_head,
                    dropout=global_config.dropout,
                    mix_type=model_config.gnn_att_mix_type,
                )
                for block_type in model_config.aa_encoder_block.split('-')
            ]
        )
        self.chunk_size = global_config.chunk_size
        self.x_pooling_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(model_config.aa_node_hidden, model_config.node_hidden),
            torch.nn.LeakyReLU(),
        )
        self.edge_pooling_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(model_config.aa_edge_hidden, model_config.edge_hidden),
            torch.nn.LeakyReLU(),
        )

        if model_config.encoder_scaf_bias:
            self.x_in = torch.nn.Embedding(model_config.n_scaf, model_config.node_hidden)
            self.edge_in = torch.nn.Embedding(
                global_config.ppl_force_max_atom_with_ring ** 2 + 2,
                model_config.edge_hidden)  # +2: BLANK edge and mask token

        self.encoder_blocks = RNNGNNBlock(
            RNN_hidden=model_config.RNN_hidden,
            GNN_block_type=model_config.encoder_block,
            node_hidden=model_config.node_hidden,
            edge_hidden=model_config.edge_hidden,
            n_head=model_config.n_head,
            dropout=global_config.dropout,
            mix_type=model_config.gnn_att_mix_type,
            chunk_size=global_config.chunk_size,
        )
        self.encoder_start = torch.nn.Parameter(torch.zeros(model_config.RNN_hidden))
        self.encoder_end = torch.nn.Parameter(torch.zeros(model_config.RNN_hidden))

        if self.model_config.with_aux:
            fp_all_hidden = model_config.fp_mogan_hidden + model_config.fp_MACCS_hidden
            self.fp_mlp = torch.nn.Sequential(
                torch.nn.Linear(fp_all_hidden, fp_all_hidden),
                torch.nn.Dropout(global_config.dropout),
                torch.nn.LeakyReLU(),
                torch.nn.LayerNorm(fp_all_hidden),
                torch.nn.Linear(fp_all_hidden, fp_all_hidden),
                torch.nn.LeakyReLU(),
            )

        if self.model_config.with_aux:
            pooling_hidden = model_config.RNN_hidden + fp_all_hidden
            self.graph_to_noise = torch.nn.Sequential(
                torch.nn.LayerNorm(pooling_hidden),
                torch.nn.Linear(pooling_hidden, pooling_hidden),
                torch.nn.Dropout(global_config.dropout),
                torch.nn.LeakyReLU(),
                torch.nn.LayerNorm(pooling_hidden),
                torch.nn.Linear(pooling_hidden, model_config.noise_hidden * 2),
            )
        else:
            pooling_hidden = model_config.RNN_hidden
            self.graph_to_noise = torch.nn.Sequential(
                torch.nn.LayerNorm(pooling_hidden),
                torch.nn.Linear(pooling_hidden, pooling_hidden),
                torch.nn.Dropout(global_config.dropout),
                torch.nn.LeakyReLU(),
                torch.nn.LayerNorm(pooling_hidden),
                torch.nn.Linear(pooling_hidden, model_config.noise_hidden * 2),
            )

        self.ppl_force_max_atom_with_ring = global_config.ppl_force_max_atom_with_ring

    def forward(self, dic_data):
        dic_data['pooling_mask'] = (dic_data['scaf_idx_bfs'] > 0).float()  # zero for padding

        dic_output = self.train_encoder(
            pad_mask(dic_data['l_x_init']),
            pad_mask(dic_data['l_edge_init']),
            dic_data,
        )
        dic_data.update(dic_output)

        if self.model_config.with_aux:
            i = 1
            with torch.no_grad():
                dic_output = self.train_encoder(
                    dic_data[f'l_x_init_masked_{i}'],
                    dic_data[f'l_edge_init_masked_{i}'],
                    dic_data,
                    atom_and_bond_mask_posi=(
                        dic_data[f'l_x_init_masked_{i}'][..., -1], dic_data[f'l_edge_init_masked_{i}'][..., -1]
                    ),
                    use_fp_dropout=True,
                )
            dic_data[f'noise_mean_mask_{i}'] = dic_output['noise_mean']
            dic_data[f'noise_logvar_mask_{i}'] = dic_output['noise_logvar']
            dic_data[f'noise_mask_{i}'] = dic_output['noise']

            i = 2
            dic_output = self.train_encoder(
                dic_data[f'l_x_init_masked_{i}'],
                dic_data[f'l_edge_init_masked_{i}'],
                dic_data,
                atom_and_bond_mask_posi=(
                    dic_data[f'l_x_init_masked_{i}'][..., -1], dic_data[f'l_edge_init_masked_{i}'][..., -1]
                ),
                use_fp_dropout=True,
            )
            dic_data[f'noise_mean_mask_{i}'] = dic_output['noise_mean']
            dic_data[f'noise_logvar_mask_{i}'] = dic_output['noise_logvar']
            dic_data[f'noise_mask_{i}'] = dic_output['noise']

        return dic_data

    def train_encoder(
            self,
            x_aa_init, edge_aa_init, dic_data,
            atom_and_bond_mask_posi=None,
            use_fp_dropout=False,
    ):
        if atom_and_bond_mask_posi is not None:
            frag_mask = self.get_frag_mask(
                dic_data, atom_and_bond_mask_posi)
        else:
            frag_mask = None

        dic_data = self.train_encoder_aa(x_aa_init, edge_aa_init, dic_data)
        dic_data = self.train_encoder_frag(dic_data, frag_mask=frag_mask)
        dic_output = self.pooling_to_noise(dic_data, use_fp_dropout=use_fp_dropout)

        return dic_output

    def get_frag_mask(self, dic_data, atom_and_bond_mask_posi):
        atom_mask_posi, bond_mask_posi = atom_and_bond_mask_posi

        component_idx = dic_data['component_idx']
        x_frag_mask = scatter_add(atom_mask_posi, component_idx, dim=-1).clamp(max=1.0)
        edge_frag_mask = scatter_add(
            bond_mask_posi,
            repeat(component_idx, 'b j -> b i j', i=component_idx.shape[-1]),
            dim=-1,
        )
        edge_frag_mask = scatter_add(edge_frag_mask, component_idx, dim=-2).clamp(max=1.0)

        x_frag_mask_bfs = torch.gather(
            x_frag_mask,
            index=dic_data['bfs_idx'],
            dim=1,
        )
        edge_frag_mask_bfs = torch.gather(
            edge_frag_mask,
            index=repeat(
                dic_data['bfs_idx'], 'b i -> b i j',
                j=edge_frag_mask.shape[2],),
            dim=1,
        )
        edge_frag_mask_bfs = torch.gather(
            edge_frag_mask_bfs,
            index=repeat(
                dic_data['bfs_idx'], 'b j -> b i j',
                i=edge_frag_mask_bfs.shape[1],),
            dim=2,
        )

        return x_frag_mask_bfs, edge_frag_mask_bfs

    def train_encoder_aa(self, x_aa_init, edge_aa_init, dic_data):
        x_aa, edge_aa = self.encode_aa(x_aa_init, edge_aa_init, dic_data['l_mask'])

        x_pooling, edge_pooling = self.scatter_pooling(x_aa, edge_aa, dic_data['component_idx'])
        # x_pooling and edge_pooling may larger than max_frag_size due to padding
        x_pooling = self.x_pooling_embed_layer(x_pooling)
        edge_pooling = self.edge_pooling_embed_layer(edge_pooling)
        x_pooling_bfs = torch.gather(
            x_pooling,
            index=repeat(dic_data['bfs_idx'], 'b n -> b n d', d=x_pooling.shape[-1]),
            dim=1,
        )
        edge_pooling_bfs = torch.gather(
            edge_pooling,
            index=repeat(
                dic_data['bfs_idx'], 'b i -> b i j d',
                j=edge_pooling.shape[-2], d=edge_pooling.shape[-1]),
            dim=1,
        )
        edge_pooling_bfs = torch.gather(
            edge_pooling_bfs,
            index=repeat(
                dic_data['bfs_idx'], 'b j -> b i j d',
                i=edge_pooling_bfs.shape[-3], d=edge_pooling.shape[-1]),
            dim=2,
        )

        dic_data['x_pooling_bfs'] = x_pooling_bfs
        dic_data['edge_pooling_bfs'] = edge_pooling_bfs

        return dic_data

    def train_encoder_frag(self, dic_data, frag_mask=None, use_fp_dropout=False):
        if frag_mask is not None:
            x_frag_mask_bfs, edge_frag_mask_bfs = frag_mask
        else:
            device = dic_data['x_pooling_bfs'].device
            x_frag_mask_bfs = torch.zeros(dic_data['x_pooling_bfs'].shape[:-1], device=device)
            edge_frag_mask_bfs = torch.zeros(dic_data['edge_pooling_bfs'].shape[:-1], device=device)

        # pooling and sorted with bfs
        x_pooling_bfs = dic_data['x_pooling_bfs']
        edge_pooling_bfs = dic_data['edge_pooling_bfs']

        # pooling
        x_init_idx = dic_data['scaf_idx_bfs']
        edge_init_idx = dic_data['reordered_scaf_sparse_adj_bfs']
        pooling_mask = dic_data['pooling_mask']

        B, N = x_init_idx.shape[:2]
        if self.model_config.encoder_scaf_bias:
            x_init = self.x_in(x_init_idx) + x_pooling_bfs
            edge_init = self.edge_in(edge_init_idx) + edge_pooling_bfs
        else:
            x_init = x_pooling_bfs
            edge_init = edge_pooling_bfs

        x_init = x_init * (1 - x_frag_mask_bfs.unsqueeze(dim=-1))
        edge_init = edge_init * (1 - edge_frag_mask_bfs.unsqueeze(dim=-1))

        h = repeat(self.encoder_start, 'd -> b d', b=B)

        for i in range(N)[::-1]:
            mask = torch.zeros(B, N, device=x_init.device)
            mask[:, :i + 1] = 1
            _, _, new_h = self.encoder_blocks(x_init, edge_init, h, mask=mask)
            h_mask = pooling_mask[:, i].unsqueeze(dim=-1)
            h = new_h * h_mask + h * (1 - h_mask)

        end = repeat(self.encoder_end, 'd -> b d', b=B)
        for i in range(len(self.encoder_blocks.GRU_cells)):
            h = self.encoder_blocks.dropout(self.encoder_blocks.GRU_cells[i](end, h))

        dic_data['pooling'] = h

        return dic_data

    def pooling_to_noise(self, dic_data, use_fp_dropout=False):
        if self.model_config.with_aux:
            fp = dic_data['fp']
            if use_fp_dropout:
                fp = F.dropout(fp, p=self.model_config.fp_dropout, training=True)  # training always True

            g = torch.concat([
                self.fp_mlp(fp), dic_data['pooling'],
            ], dim=-1)
        else:
            g = dic_data['pooling']

        noise_mean, noise_logvar = self.graph_to_noise(g).chunk(2, dim=-1)
        noise = self.reparametrize(noise_mean, noise_logvar)

        dic_output = dict(
            noise_mean=noise_mean,
            noise_logvar=noise_logvar,
            noise=noise,
        )

        return dic_output

    def encode_aa(self, x, edge, mask):
        x = self.l_node_embed_layer(x)
        edge = self.l_edge_embed_layer(edge)

        for i, block in enumerate(self.aa_blocks):
            x, edge = block(x, edge, mask=mask, chunk_size=self.chunk_size)

        return x, edge

    def scatter_pooling(self, x, edge, component_idx):
        x_pooling_mean = scatter_mean(x, component_idx, dim=-2)

        edge_pooling_mean = scatter_mean(
            edge,
            repeat(component_idx, 'b j -> b i j', i=component_idx.shape[-1]),
            dim=-2,
        )
        edge_pooling_mean = scatter_mean(edge_pooling_mean, component_idx, dim=-3)

        # x_pooling_max = scatter_max(x, component_idx, dim=-2)[0]
        #
        # edge_pooling_max = scatter_max(
        #     edge,
        #     repeat(component_idx, 'b j -> b i j', i=component_idx.shape[-1]),
        #     dim=-2,
        # )[0]
        # edge_pooling_max = scatter_max(edge_pooling_max, component_idx, dim=-3)[0]

        x_pooling = torch.concat([
            x_pooling_mean,
            # x_pooling_max,
        ], dim=-1)
        edge_pooling = torch.concat([
            edge_pooling_mean,
            # edge_pooling_max,
        ], dim=-1)

        return x_pooling, edge_pooling

    def reparametrize(self, noise_mean, noise_logvar):
        # N(mu, std^2) = N(0, 1) * std + mu
        noise = torch.randn_like(noise_logvar).mul(
            (0.5 * noise_logvar).exp()
        ).add_(
            noise_mean
        )
        return noise

    def flat_graph(self, x, edge, mask):
        x = self.reudce_x_layer(x)
        edge = self.reudce_edge_layer(edge)

        x = x * mask.unsqueeze(dim=-1)
        pair_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-2)
        edge = edge * pair_mask.unsqueeze(dim=-1)

        x = F.pad(
            x,
            (0, 0, 0, global_config.max_l_len - x.shape[1]),
            'constant',
            0,
        )
        edge = F.pad(
            edge,
            (
                0, 0,
                0, global_config.max_l_len - edge.shape[1],
                0, global_config.max_l_len - edge.shape[1],
            ),
            'constant',
            0,
        )

        x = rearrange(x, 'b n d -> b (n d)')
        edge = rearrange(edge, 'b i j d -> b (i j d)')

        flat_pooling = self.graph_flat_layer(torch.concat([x, edge], dim=-1))

        return flat_pooling


class ScafDecoder(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        self.x_in = torch.nn.Embedding(model_config.n_scaf, model_config.node_hidden)

        self.noise_to_graph = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.noise_hidden),
            torch.nn.Linear(model_config.noise_hidden, model_config.RNN_hidden),
        )

        self.decoder = RNNGNNBlock(
            model_config.RNN_hidden,
            model_config.decoder_block,
            model_config.node_hidden,
            model_config.edge_hidden,
            model_config.n_head,
            dropout=global_config.dropout,
            mix_type=model_config.gnn_att_mix_type,
            chunk_size=global_config.chunk_size,
        )
        self.decoder_start = torch.nn.Parameter(torch.zeros(model_config.RNN_hidden))

        self.x2edge_left_in = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.RNN_hidden),
            torch.nn.Linear(model_config.RNN_hidden, model_config.RNN_hidden // 2),
        )
        self.x2edge_right_in = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.RNN_hidden),
            torch.nn.Linear(model_config.RNN_hidden, model_config.RNN_hidden // 2),
        )
        self.x2edge_out = torch.nn.Linear(model_config.RNN_hidden, model_config.edge_hidden)

        edge_post_hidden = model_config.RNN_hidden * 4 + model_config.edge_hidden
        self.lin_edge = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_post_hidden),
            torch.nn.Linear(edge_post_hidden, model_config.RNN_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(model_config.RNN_hidden),
            torch.nn.Linear(model_config.RNN_hidden, model_config.RNN_hidden),
            torch.nn.LeakyReLU(),
        )

        self.x_out = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.RNN_hidden),
            torch.nn.Linear(model_config.RNN_hidden, model_config.n_scaf),
        )
        self.edge_out_frag = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.RNN_hidden),
            torch.nn.Linear(model_config.RNN_hidden, 1),
        )
        self.edge_out_bond = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.RNN_hidden),
            torch.nn.Linear(model_config.RNN_hidden, global_config.ppl_force_max_atom_with_ring ** 2 + 2),
        )

        aa_x_in_hidden = model_config.RNN_hidden + model_config.node_hidden + global_config.l_label_node_hidden + 1
        self.aa_x_in = torch.nn.Sequential(
            torch.nn.LayerNorm(aa_x_in_hidden),
            torch.nn.Linear(aa_x_in_hidden, model_config.aa_node_hidden),
            torch.nn.LeakyReLU(),
        )
        self.register_buffer('masked_aa_feat', torch.tensor([0] * global_config.l_label_node_hidden + [1], dtype=torch.float))
        self.aa_edge_in = torch.nn.Sequential(
            torch.nn.LayerNorm(global_config.l_label_edge_hidden),
            torch.nn.Linear(global_config.l_label_edge_hidden, model_config.aa_edge_hidden),
            torch.nn.LeakyReLU(),
        )
        self.aa_blocks = torch.nn.ModuleList(
            [
                GNNBlock(
                    block_type=block_type,
                    node_hidden=model_config.aa_node_hidden,
                    edge_hidden=model_config.aa_edge_hidden,
                    n_head=model_config.n_head,
                    dropout=model_config.aa_decoder_dropout,
                    mix_type=model_config.gnn_att_mix_type,
                )
                for block_type in model_config.aa_decoder_block.split('-')
            ]
        )
        self.chunk_size = global_config.chunk_size
        self.aa_x_out = torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.aa_node_hidden + model_config.aa_edge_hidden),
            torch.nn.Linear(model_config.aa_node_hidden + model_config.aa_edge_hidden,
                            global_config.l_label_node_hidden),
        )

        self.ppl_force_max_atom_with_ring = global_config.ppl_force_max_atom_with_ring

        self.register_buffer('frag_size_pool', torch.empty(model_config.frag_size_pool).fill_(-1))
        self.df_scaf = pd.read_csv(self.model_config.frag_codebook_path)
        self.register_buffer(
            'dic_frag_size',
            F.pad(
                torch.from_numpy(self.df_scaf['frag_size'].values),
                (1, 3), 'constant', global_config.ppl_force_max_atom_with_ring,
            )
        )
        self.register_buffer(
            'dic_att_num',
            F.pad(
                torch.from_numpy(np.load(self.model_config.scaf_att_num_path)['all_att_num']),
                (0, 0, 1, 3), 'constant', 5,
            )
        )

    def forward(self, dic_data):
        dic_data = self.train_decoder(dic_data)
        dic_data = self.train_decoder_aa(dic_data)

        return dic_data

    def train_decoder(self, dic_data):
        x_init = dic_data['scaf_idx_bfs']
        pooling_mask = dic_data['pooling_mask']
        B, N = x_init.shape[:2]
        device = x_init.device
        dic_output = defaultdict(list)

        h = self.noise_to_graph(dic_data['noise'])

        edge = torch.zeros(B, N, N, self.model_config.edge_hidden, device=device, dtype=h.dtype)
        for i in range(N):
            if i == 0:
                start = repeat(self.decoder_start, 'd -> b d', b=B)
                for gru_i in range(len(self.decoder.GRU_cells)):
                    h = self.decoder.dropout(self.decoder.GRU_cells[gru_i](start, h))

                edge[:, [0], 0] = self.x2edge(h.unsqueeze(dim=1), h)
            else:
                _, new_edge, new_h = self.decoder(
                    self.x_in(x_init[:, :i]), edge[:, :i, :i], h, mask=pooling_mask[:, :i])

                sub_edge = torch.zeros(B, N, N, self.model_config.edge_hidden, device=device)
                sub_edge[:, :i, :i] = new_edge
                sub_edge[:, :i, i] = self.x2edge(
                    torch.stack(dic_output['graph_hidden'], dim=1)[:, :i],
                    new_h,
                )
                sub_mask = rearrange(pooling_mask[:, i], 'b -> b () () ()')
                edge = sub_edge * sub_mask + edge * (1 - sub_mask)

                h_mask = pooling_mask[:, i].unsqueeze(dim=-1)
                h = new_h * h_mask + h * (1 - h_mask)

            dic_output['graph_hidden'].append(h)

        # extra: for end token prediction
        x = self.x_in(x_init)
        x, edge, h = self.decoder(x, edge, h, mask=pooling_mask)
        dic_output['graph_hidden'].append(h)

        dic_output = {k: torch.stack(v, dim=0) for k, v in dic_output.items()}
        dic_data.update(dic_output)

        dic_data['x_scaf_pred'] = self.x_out(dic_data['graph_hidden'])
        dic_data['scaf_idx_bfs_with_end_token'] = F.pad(
            dic_data['scaf_idx_bfs'],
            (0, 1), 'constant', self.model_config.x_end_token)

        # for edge
        edge = self.lin_edge(torch.concat([
            repeat(dic_output['graph_hidden'][:x_init.shape[1]], 'i b d -> b i j d', j=x_init.shape[1]),
            repeat(dic_output['graph_hidden'][:x_init.shape[1]], 'j b d -> b i j d', i=x_init.shape[1]),
            repeat(self.x_in(x_init), 'b i d -> b i j d', j=x_init.shape[1]),
            repeat(self.x_in(x_init), 'b j d -> b i j d', i=x_init.shape[1]),
            edge,
        ], dim=-1))
        dic_data['edge_scaf_pred'] = self.edge_out_frag(edge).squeeze(dim=-1)
        dic_data['edge_bond_pred'] = self.edge_out_bond(edge)

        dic_data['edge_allow_mask'] = self.get_allow_edge(
            dic_data['scaf_idx_bfs'],
            torch.empty(B, N, N, device=device, dtype=torch.long).fill_(self.model_config.edge_mask_token),
            pooling_mask,
        )

        return dic_data

    def x2edge(self, left, right):
        # left: [B, N, D]
        # right: [B, D]
        left = self.x2edge_left_in(left)
        right = self.x2edge_right_in(right)

        right = repeat(right, 'b d -> b n d', n=left.shape[1])
        edge = torch.concat(
            [
                left * right,
                left - right,
            ], dim=-1,
        )
        edge = self.x2edge_out(edge)

        return edge

    def train_decoder_aa(self, dic_data):
        x_aa_init = dic_data['l_x_init']
        edge_aa_init = dic_data['l_edge_init']

        x_aa_masked, dic_data['x_aa_masked_posi'] = self.uni_mask_aa(x_aa_init)

        graph_hidden = torch.gather(
            rearrange(dic_data['graph_hidden'], 'n b d -> b n d'),
            index=repeat(dic_data['bfs_idx_reverse'], 'b n -> b n d', d=dic_data['graph_hidden'].shape[-1]), dim=1,
        )
        x_scaf = torch.concat([self.x_in(dic_data['scaf_idx']), graph_hidden], dim=-1)
        x_aa_from_scaf = self.unpooling(x_scaf, dic_data['component_idx'])

        x_aa_in = torch.concat([
            x_aa_masked, x_aa_from_scaf
        ], dim=-1)
        x_aa_pred = self.pred_aa(x_aa_in, edge_aa_init, dic_data['l_mask'])

        dic_data['x_aa_pred'] = x_aa_pred

        return dic_data

    def pred_aa(self, x_aa_in, edge_aa_init, mask):
        x_aa = self.aa_x_in(x_aa_in)
        edge_aa = self.aa_edge_in(edge_aa_init)
        for i, block in enumerate(self.aa_blocks):
            x_aa, edge_aa = block(x_aa, edge_aa, mask=mask, chunk_size=self.chunk_size)
        x_aa_pred = self.aa_x_out(
            torch.concat(
                [
                    x_aa,
                    (edge_aa * rearrange(mask, 'b j -> b () j ()')).sum(dim=-2) /
                    reduce(mask, 'b j -> b () ()', 'sum')
                ], dim=-1
            )
        )
        return x_aa_pred

    @torch.no_grad()
    def uni_mask_aa(self, x):
        batch_mask_rate = torch.rand(x.shape[0], device=x.device)
        rand_x = torch.rand(x.shape[:2], device=x.device)
        mask_posi = rand_x < batch_mask_rate.unsqueeze(dim=-1)

        masked_x = torch.empty_like(x).copy_(x)
        masked_x = F.pad(masked_x, (0, 1), 'constant', 0)
        masked_x[mask_posi] = self.masked_aa_feat

        return masked_x, mask_posi

    def unpooling(self, x, component_idx, is_mask=False):
        if is_mask:
            x = F.pad(x, (0, component_idx.max()+1 - x.shape[1]), 'constant', 0)  # padding
            x = torch.gather(x, index=component_idx, dim=-1)
        else:
            x = F.pad(x, (0, 0, 0, component_idx.max() + 1 - x.shape[1]), 'constant', 0)  # padding
            x = torch.gather(x, index=repeat(component_idx, 'b i -> b i d', d=x.shape[-1]), dim=-2)
        return x

    @torch.no_grad()
    def sample(
        self,
        batch_size=32,
        max_frag_size=None,
        input_noise=None,
        output_smi=True,
        rescue=True,
        vocab_mask=None,
    ):
        # ================ frag ================
        device = self.x_in.weight.device
        if max_frag_size is None:
            max_frag_size = self.model_config.max_frag_size

        if input_noise is not None:
            noise = input_noise
        else:
            noise = torch.randn(batch_size, self.model_config.noise_hidden, device=device)
        h = self.noise_to_graph(noise)

        B, N = h.shape[0], max_frag_size
        dic_output = defaultdict(list)

        edge = torch.zeros(B, N, N, self.model_config.edge_hidden, device=device, dtype=h.dtype)
        pooling_mask = torch.ones(B, N, device=device)

        for i in range(max_frag_size):
            if i == 0:
                start = repeat(self.decoder_start, 'd -> b d', b=B)
                for gru_i in range(len(self.decoder.GRU_cells)):
                    h = self.decoder.GRU_cells[gru_i](start, h)
                dic_output['graph_hidden'].append(h)

                x_pred = self.x_out(h)
                if vocab_mask is not None:
                    x_pred = x_pred + vocab_mask
                x_pred = self.control_x(x_pred)
                x_pred[:, self.model_config.x_end_token] = -get_max_tensor(x_pred)

                sampled_new_x = x_pred.argmax(dim=-1)
                dic_output['sampled_x'].append(sampled_new_x)

                edge[:, [0], 0] = self.x2edge(h.unsqueeze(dim=1), h)
            else:
                x_init = torch.stack(dic_output['sampled_x'], dim=1)
                _, new_edge, new_h = self.decoder(
                    self.x_in(x_init), edge[:, :i, :i], h, mask=pooling_mask[:, :i])

                x_pred = self.x_out(new_h)
                if vocab_mask is not None:
                    x_pred = x_pred + vocab_mask
                x_pred = self.control_x(x_pred)
                if i == self.model_config.max_frag_size:
                    x_pred[:, self.model_config.x_end_token] = get_max_tensor(x_pred)
                sampled_new_x = x_pred.argmax(dim=-1)
                dic_output['sampled_x'].append(sampled_new_x)

                pooling_mask[:, i] = pooling_mask[:, i - 1] * (sampled_new_x != self.model_config.x_end_token)

                sub_edge = torch.zeros(B, N, N, self.model_config.edge_hidden, device=device)
                sub_edge[:, :i, :i] = new_edge
                sub_edge[:, :i, i] = self.x2edge(
                    torch.stack(dic_output['graph_hidden'], dim=1)[:, :i],
                    new_h,
                )
                sub_mask = rearrange(pooling_mask[:, i], 'b -> b () () ()')
                edge = sub_edge * sub_mask + edge * (1 - sub_mask)

                h_mask = pooling_mask[:, i].unsqueeze(dim=-1)
                h = new_h * h_mask + h * (1 - h_mask)
                dic_output['graph_hidden'].append(h)

        # extra
        x = self.x_in(torch.stack(dic_output['sampled_x'], dim=1))
        _, edge, h = self.decoder(x, edge, h, mask=pooling_mask)
        dic_output['graph_hidden'].append(h)

        x_init = torch.stack(dic_output['sampled_x'], dim=1)

        # ================ edge ================
        edge = self.lin_edge(torch.concat([
            repeat(
                torch.stack(dic_output['graph_hidden'][:x_init.shape[1]], dim=0),
                'i b d -> b i j d', j=x_init.shape[1]),
            repeat(
                torch.stack(dic_output['graph_hidden'][:x_init.shape[1]], dim=0),
                'j b d -> b i j d', i=x_init.shape[1]),
            repeat(self.x_in(x_init), 'b i d -> b i j d', j=x_init.shape[1]),
            repeat(self.x_in(x_init), 'b j d -> b i j d', i=x_init.shape[1]),
            edge,
        ], dim=-1))
        edge_scaf_pred = self.edge_out_frag(edge).squeeze(dim=-1)
        edge_bond_pred = self.edge_out_bond(edge)

        x_init.masked_fill_(~pooling_mask.bool(), self.model_config.x_mask_token)
        edge_init = torch.empty(B, N, N, device=device, dtype=torch.long).fill_(self.model_config.edge_mask_token)

        edge_scaf_pred.masked_fill_(
            torch.ones_like(edge_scaf_pred).tril(diagonal=0).bool(), -get_max_tensor(edge_scaf_pred))

        for i in range(N):
            # select bridge posi
            if i > 1:
                # no ring between frags
                for batch_i in range(B):
                    for _ in range(i):
                        sub_edge_init = torch.empty_like(edge_init[batch_i]).copy_(edge_init[batch_i])
                        sub_edge_scaf = (
                                sub_edge_init[:i+1, :i+1] < (self.model_config.ppl_force_max_atom_with_ring ** 2)
                        ).long()
                        sele_j = edge_scaf_pred[batch_i, :, i].argmax(dim=-1)
                        sub_edge_scaf[sele_j, i] = 1
                        if has_cycle(sub_edge_scaf):
                            edge_scaf_pred[batch_i, sele_j, i] = -get_max_tensor(edge_scaf_pred)
                        else:
                            break
            bridge_posi = edge_scaf_pred.argmax(dim=-2)

            # select bridge type
            cur_edge_pred = torch.empty_like(edge_bond_pred).copy_(edge_bond_pred)
            cur_edge_pred = self.control_edge(x_init, edge_init, pooling_mask, cur_edge_pred)
            sampled_new_edge = cur_edge_pred.argmax(dim=-1)
            for batch_i in range(B):
                j = int(bridge_posi[batch_i, i])
                edge_init[batch_i, j, i] = sampled_new_edge[batch_i, j, i]
            edge_init = self.sym_edge(edge_init, sym_sparse_edge=True)

        # ================ atom ================
        edge_aa_init, component_idx = self.gen_scaf(
            x_init,
            edge_init,
            pooling_mask,
        )
        B_aa, N_aa = edge_aa_init.shape[:2]
        x_aa_init = repeat(self.masked_aa_feat, 'd -> b n d', b=B_aa, n=N_aa)
        aa_mask = self.unpooling(pooling_mask, component_idx, is_mask=True)

        x_scaf = torch.concat([self.x_in(x_init), torch.stack(dic_output['graph_hidden'][:x_init.shape[1]], dim=1)], dim=-1)
        x_aa_from_scaf = self.unpooling(x_scaf, component_idx)

        for i in range(N_aa):
            x_aa_in = torch.concat([
                x_aa_init, x_aa_from_scaf
            ], dim=-1)

            x_aa_pred = self.pred_aa(x_aa_in, edge_aa_init, aa_mask)

            masked_posi = x_aa_init[..., -1] == 1
            rand = torch.rand_like(masked_posi, dtype=torch.float)
            rand.masked_fill_(~masked_posi, -get_max_tensor(rand))
            unmasked_posi = (reduce(rand, 'b n -> b ()', 'max') == rand).unsqueeze(dim=-1).float()

            x_aa_pred = F.one_hot(x_aa_pred.argmax(dim=-1), num_classes=x_aa_init.shape[-1])
            x_aa_init = x_aa_init * (1 - unmasked_posi) + x_aa_pred * unmasked_posi

        x_aa = x_aa_init.argmax(dim=-1)
        edge_aa = edge_aa_init.argmax(dim=-1)
        x_scaf = x_init

        dic_output = dict(
            x_aa=x_aa,
            edge_aa=edge_aa,
            pooling_mask=pooling_mask,
            aa_mask=aa_mask,
            component_idx=component_idx,
            x_scaf=x_scaf,
            edge_scaf=edge_init,
        )

        if output_smi:
            smi_list = []
            for batch_i in range(B):
                aa_mask = dic_output['aa_mask'][batch_i].bool()
                smi = reconst_smi(
                    dic_output['edge_aa'][batch_i][aa_mask, :][:, aa_mask],
                    x_aa=dic_output['x_aa'][batch_i][aa_mask],
                    rescue=rescue,
                )
                smi_list.append(smi)

            dic_output['smi'] = smi_list

        return dic_output

    @torch.no_grad()
    def gen_scaf(self, x, edge, mask):
        edge_aa_init_list = []
        component_idx_list = []
        n_max = 0
        for i in range(x.shape[0]):
            edge_aa_init, component_idx = combine_scaf(
                x[i],
                edge[i],
                mask[i],
                global_config.ppl_force_max_atom_with_ring,
                self.df_scaf,
                scaf_feat_path=None,
            )
            edge_aa_init_list.append(edge_aa_init)
            component_idx_list.append(component_idx)
            n_max = max(len(component_idx), n_max)

        edge_aa_init_pad = []
        component_idx_pad = []
        for i in range(x.shape[0]):
            edge_aa_init = edge_aa_init_list[i]
            component_idx = component_idx_list[i]

            n_add = n_max - edge_aa_init.shape[0]
            edge_aa_init = F.pad(edge_aa_init, (0, 0, 0, n_add, 0, n_add), 'constant', 0)
            component_idx = F.pad(component_idx, (0, n_add), 'constant', component_idx.max() + 1)

            edge_aa_init_pad.append(edge_aa_init)
            component_idx_pad.append(component_idx)
        edge_aa_init = torch.stack(edge_aa_init_pad, dim=0)
        component_idx = torch.stack(component_idx_pad, dim=0)

        return edge_aa_init, component_idx

    @torch.no_grad()
    def control_x(self, x_pred):
        x_pred[..., [0, self.model_config.x_start_token, self.model_config.x_mask_token]] = -get_max_tensor(x_pred)
        return x_pred

    def get_allow_edge(self, x, edge, mask):
        # control with size
        # do not allow edge type mismatch with frag size
        x_size = self.dic_frag_size[x]
        allow_with_size_x = (
                    torch.arange(self.ppl_force_max_atom_with_ring, device=x.device) < x_size.unsqueeze(dim=-1)).float()
        allow_with_size_pair = rearrange(allow_with_size_x, 'b i p -> b i () p ()') * \
                               rearrange(allow_with_size_x, 'b j q -> b () j () q')
        pair_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-2)
        allow_with_size_pair = allow_with_size_pair * rearrange(pair_mask, 'b i j -> b i j () ()')

        allow_with_size = F.pad(
            rearrange(allow_with_size_pair, 'b i j p q -> b i j (p q)'),
            (0, 2), 'constant', 1,
        )  # add BLANK and mask
        allow_with_size[..., -1] = 0  # no mask

        # control att
        x_att_allow_num = self.dic_att_num[x]
        edge_ont_hot = F.one_hot(edge, num_classes=self.ppl_force_max_atom_with_ring ** 2 + 2)
        current_att_hit = rearrange(
            edge_ont_hot[..., :self.ppl_force_max_atom_with_ring ** 2],
            'b i j (p q) -> b i j p q', p=self.ppl_force_max_atom_with_ring,
        ) * allow_with_size_pair
        current_att_count = reduce(
            current_att_hit,
            'b i j p q -> b i p', 'sum',
        )
        x_allow_att = (x_att_allow_num - current_att_count) > 0  # have at least one att remain

        allow_with_att_num = rearrange(x_allow_att, 'b i p -> b i () p ()') * \
                             rearrange(x_allow_att, 'b j q -> b () j () q')
        allow_with_att_num = F.pad(
            rearrange(allow_with_att_num, 'b i j p q -> b i j (p q)'),
            (0, 2), 'constant', 1,
        )  # add BLANK and mask
        allow_with_att_num[..., -1] = 0  # no mask

        allow_mask = allow_with_size * allow_with_att_num
        return allow_mask  # [B, N, N, n_edge_type]

    @torch.no_grad()
    def control_edge(self, x, edge, mask, edge_pred):
        allow_mask = self.get_allow_edge(x, edge, mask)

        # diag is BLANK
        B, N, _, D = allow_mask.shape
        diag = repeat(torch.eye(N, device=allow_mask.device), 'i j -> b i j', b=B)
        pair_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-2)
        diag = diag * pair_mask
        blank = torch.zeros(D, device=allow_mask.device)
        blank[-2] = 1
        allow_mask[diag.bool()] = blank

        # control edge prediction
        edge_pred.masked_fill_(~allow_mask.bool(), -get_max_tensor(edge_pred))

        return edge_pred

    def sym_edge(self, edge, sym_sparse_edge=True):
        triu_mask = torch.ones(edge.shape[1], edge.shape[1], device=edge.device, dtype=torch.long).triu(diagonal=0)
        if sym_sparse_edge:
            edge_ext = F.one_hot(edge, num_classes=self.ppl_force_max_atom_with_ring ** 2 + 2)

            # sym mask
            triu_mask = triu_mask.unsqueeze(dim=-1)
            edge_ext = edge_ext * triu_mask + rearrange(edge_ext, 'b i j d -> b j i d') * (1 - triu_mask)

            # sym bond
            edge_ext_only_bond = rearrange(
                edge_ext[..., :self.ppl_force_max_atom_with_ring ** 2],
                'b i j (p q) -> b i j p q', p=self.ppl_force_max_atom_with_ring,
            )
            triu_mask = triu_mask.unsqueeze(dim=-1)
            edge_ext_only_bond = edge_ext_only_bond * triu_mask + \
                                 rearrange(edge_ext_only_bond, 'b i j p q -> b j i q p') * (1 - triu_mask)
            edge_ext_only_bond = rearrange(edge_ext_only_bond, 'b i j p q -> b i j (p q)')
            edge_ext[..., :self.ppl_force_max_atom_with_ring ** 2] = edge_ext_only_bond
            edge = edge_ext.argmax(dim=-1)
        else:
            edge = edge * triu_mask + rearrange(edge, 'b i j -> b j i') * (1 - triu_mask)
        return edge


class ScafVAEBase(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        model_config = ScafVAE_config if model_config is None else model_config
        self.model_config = model_config

        self.frag_encoder = ScafEncoder(model_config)
        self.frag_decoder = ScafDecoder(model_config)

        if self.model_config.with_aux:
            if model_config.shallow_aux:
                self.aux_proj_cl = torch.nn.Linear(model_config.noise_hidden, model_config.cl_hidden)
                fp_all_hidden = model_config.fp_mogan_hidden + model_config.fp_MACCS_hidden
                self.aux_proj_fp = torch.nn.Linear(model_config.noise_hidden, fp_all_hidden)
            else:
                self.aux_proj_cl = torch.nn.Sequential(
                    torch.nn.LayerNorm(model_config.noise_hidden),
                    torch.nn.Linear(model_config.noise_hidden, model_config.cl_hidden),
                    torch.nn.LeakyReLU(),
                    torch.nn.LayerNorm(model_config.cl_hidden),
                    torch.nn.Linear(model_config.cl_hidden, model_config.cl_hidden),
                )
                fp_all_hidden = model_config.fp_mogan_hidden + model_config.fp_MACCS_hidden
                self.aux_proj_fp = torch.nn.Sequential(
                    torch.nn.LayerNorm(model_config.noise_hidden),
                    torch.nn.Linear(model_config.noise_hidden, fp_all_hidden),
                    torch.nn.LeakyReLU(),
                    torch.nn.LayerNorm(fp_all_hidden),
                    torch.nn.Linear(fp_all_hidden, fp_all_hidden),
                )

        self.loss_object = ScafLossFunction(model_config=model_config)

    def forward(self, dic_data, epoch=-1):
        dic_data = self.frag_encoder(dic_data)
        dic_data = self.frag_decoder(dic_data)

        if self.model_config.with_aux:
            dic_data = self.get_cl_terms(dic_data)

        grad_loss, dic_log = self.loss_object(dic_data, epoch=epoch)

        return dic_data, grad_loss, dic_log

    def get_cl_terms(self, dic_data):
        dic_data['cl_embed_mask_1'] = self.aux_proj_cl(dic_data['noise_mask_1'])
        dic_data['cl_embed_mask_2'] = self.aux_proj_cl(dic_data['noise_mask_2'])

        dic_data['cl_embed'] = self.aux_proj_cl(dic_data['noise'])
        dic_data['fp_embed'] = self.aux_proj_fp(dic_data['noise'])

        return dic_data

    @torch.no_grad()
    def noise2repr(self, noise):
        repr = torch.concat([self.aux_proj_cl(noise), self.aux_proj_fp(noise).sigmoid()], dim=-1)
        return repr






























