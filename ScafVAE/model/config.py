from dataclasses import dataclass
import ml_collections as mlc
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import ScafVAE
from ScafVAE.utils.graph2mol import *
from ScafVAE.utils.common import *



@dataclass
class global_config:
    dropout: float = 0.1
    chunk_size: int = None  # None / int
    max_l_len: int = 128
    max_l_len_for_eval: int = 128
    max_frag_len: int = 16
    l_label_node_hidden: int = len(atom_types)  # + len(Chem.rdchem.ChiralType.values)
    l_label_edge_hidden: int = len(bond_types)

    ppl_force_max_atom: int = 6  # max number of atoms in each frag (for non-ring part)
    ppl_force_max_atom_with_ring: int = 16  # max number of atoms in each frag (with non-ring)
    vq_sim_type = 'cos'  # euclidean / cos, (only support cos now)


@dataclass
class PPLPredictor_config:
    n_head: int = 8
    n_block: int = 2
    block_type: str = 'GT'  # GT / tri
    gnn_att_mix_type: str = 'add'  # add / dot
    node_hidden: int = 128
    edge_hidden: int = 64

    gt_bond_mask_chunk: int = 64
    bond_ppl_cutoff: float = 1.5  # split mol with bond ppl
    bond_ppl_cutoff_quantile: float = None  # work when bond_ppl_cutoff = None, quantile of low to high ppl in ppl_pool
    frag_ppl_cutoff: float = None
    frag_ppl_cutoff_quantile: float = None
    ppl_force_max_atom = global_config.ppl_force_max_atom
    ppl_force_max_atom_with_ring = global_config.ppl_force_max_atom_with_ring
    min_components: int = 3
    max_split_mol_iter: int = 20


suppl_config = os.path.dirname(ScafVAE.__file__) + '/model/config.yml'
model_param_path = load_yaml(suppl_config)['model_param_path']
model_param_path = os.path.dirname(ScafVAE.__file__) + '/model_param/' if model_param_path == 'default' else model_param_path
frag_codebook_path = f'{model_param_path}/unique_scaf_max{global_config.ppl_force_max_atom}_cutoff{PPLPredictor_config.bond_ppl_cutoff}_no_ring_branch.csv'
scaf_att_num_path = f'{model_param_path}/scaf_att_num_max{global_config.ppl_force_max_atom}_cutoff{PPLPredictor_config.bond_ppl_cutoff}_no_ring_branch.npz'
if not os.path.exists(frag_codebook_path):
    print('WARNING: Can not find a bond scaffold set in model_param_path !')
    delmkdir(os.path.dirname(frag_codebook_path), remove_old=False)
    df_tmp = pd.DataFrame({'data': ['Nan']})
    df_tmp.to_csv(frag_codebook_path, index=False)



@dataclass
class ScafVAE_config:
    n_head: int = 8
    node_hidden: int = 1024
    edge_hidden: int = 64
    aa_node_hidden: int = 128
    aa_edge_hidden: int = 64
    gnn_att_mix_type: str = 'add'  # add / dot

    ppl_force_max_atom = global_config.ppl_force_max_atom
    ppl_force_max_atom_with_ring = global_config.ppl_force_max_atom_with_ring

    # [dummy/pad, ..., MASK, start, end]
    n_scaf: int = len(pd.read_csv(frag_codebook_path)) + 4
    # edge: [ppl_force_max_atom_with_ring**2, BLANK, MASK]
    x_mask_token: int = n_scaf - 3
    x_start_token: int = n_scaf - 2
    x_end_token: int = n_scaf - 1
    edge_mask_token: int = global_config.ppl_force_max_atom_with_ring ** 2 + 1

    frag_size_pool: str = int(2e+6)
    max_frag_size: int = global_config.max_frag_len

    aa_encoder_block: str = 'GT-GT'
    aa_node_hidden_reduced: int = 16
    aa_edge_hidden_reduced: int = 1
    aa_flat_hidden: int = 1024
    encoder_block: str = 'GT-GT'

    noise_hidden: int = 64
    RNN_hidden: int = 1024
    decoder_block: str = 'GT-GT'
    aa_decoder_block: str = 'GT-GT'
    aa_decoder_dropout: float = 0.0  # dropout may cause overfit in aa_decoder when noise_hidden is small

    with_aux: bool = True


@dataclass
class Aux_config:
    cl_hidden: int = 512
    fp_mogan_hidden: int = 512
    fp_MACCS_hidden: int = 167
    fp_dropout: float = 0.5


@dataclass
class TaskedModel_config:
    hidden: int = 256


training_config = mlc.ConfigDict({
    'data': {
        'mask_rate': 0.15,
        'data_loader_check_valid': False,
        'shuffle_input': False,

        'n_scaf': ScafVAE_config.n_scaf,
        'n_max_scaf': ScafVAE_config.max_frag_size,

        # Warning: max_drop_size & max_drop_retain make model hard to train
        # in collate, reduce batch size if the max frag size > max_drop_size
        'max_drop_size': None,  # global_config.max_frag_len / 38,
        # in collate, reduce batch size to max_drop_retain
        'max_drop_retain': 4
    },
    'loss': {
        'cl_T': 0.07,
    },
    'loss_weight': {
        'PPL': {
            'l_single_bond_loss': 1.0,
        },
        'FRAG': {
            'x_scaf_pred_loss': 1.0,
            'edge_scaf_pred_loss': 4.0,

            'x_aa_pred_loss': 2.0,
            'edge_bond_pred_loss': 2.0,

            'kl_loss': 0.003,

            'cl_loss': 10.0,
            'cl_embed_1_norm_penalty': 0.1,
            'cl_embed_2_norm_penalty': 0.1,

            'fp_loss': 5.0,
        },
    },
})
























