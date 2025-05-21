from dataclasses import dataclass
import ml_collections as mlc
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from openfold.config import model_config as get_af_model_config

import ScafVAE
from ScafVAE.utils.common import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.graph2mol import atom_types, bond_types


root_path = os.path.dirname(ScafVAE.__file__)
# config_yml = root_path + '/config.yml'
# config_yml = load_yaml(config_yml)


@dataclass
class global_config:
    dropout: float = 0.1
    chunk_size: int = None  # None / int
    max_l_len: int = 128
    max_l_len_for_eval: int = 128
    max_frag_size: int = 16
    l_label_node_hidden: int = len(atom_types)  # + len(Chem.rdchem.ChiralType.values)
    l_label_edge_hidden: int = len(bond_types)

    ppl_force_max_atom: int = 6  # max number of atoms in each frag (for non-ring part)
    ppl_force_max_atom_with_ring: int = 16  # max number of atoms in each frag (for ring part)


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
    bond_ppl_cutoff_quantile: float = None  # work when bond_ppl_cutoff is None, quantile of low to high ppl in ppl_pool
    frag_ppl_cutoff: float = None
    frag_ppl_cutoff_quantile: float = None
    ppl_force_max_atom = global_config.ppl_force_max_atom
    ppl_force_max_atom_with_ring = global_config.ppl_force_max_atom_with_ring
    min_components: int = 3
    max_split_mol_iter: int = 20


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

    frag_codebook_path: str = f'{root_path}/params/unique_scaf_max{global_config.ppl_force_max_atom}_cutoff{PPLPredictor_config.bond_ppl_cutoff}_no_ring_branch.csv'   # change path to your own dataset
    scaf_att_num_path: str = f'{root_path}/params/scaf_att_num_max{global_config.ppl_force_max_atom}_cutoff{PPLPredictor_config.bond_ppl_cutoff}_no_ring_branch.npz'   # change path to your own dataset

    # [dummy/pad, ..., MASK, start, end]
    n_scaf: int = len(pd.read_csv(frag_codebook_path)) + 4
    # edge: [ppl_force_max_atom_with_ring**2, BLANK, MASK]
    x_mask_token: int = n_scaf - 3
    x_start_token: int = n_scaf - 2
    x_end_token: int = n_scaf - 1
    edge_mask_token: int = global_config.ppl_force_max_atom_with_ring ** 2 + 1

    frag_size_pool: str = int(2e+6)
    max_frag_size: int = global_config.max_frag_size

    aa_encoder_block: str = 'GT-GT'
    aa_node_hidden_reduced: int = 16
    aa_edge_hidden_reduced: int = 1
    aa_flat_hidden: int = 1024
    encoder_block: str = 'GT-GT'
    encoder_scaf_bias: bool = True

    noise_hidden: int = 64
    RNN_hidden: int = 1024
    decoder_block: str = 'GT-GT'
    aa_decoder_block: str = 'GT-GT'
    aa_decoder_dropout: float = 0.0  # dropout may cause overfit in aa_decoder when noise_hidden is small

    with_aux: bool = True
    shallow_aux: bool = False
    cl_hidden: int = 512
    fp_mogan_hidden: int = 512
    fp_MACCS_hidden: int = 167
    fp_dropout: float = 0.5



training_config = mlc.ConfigDict({
    'data': {
        'mask_rate': 0.15,
        'edge_add_mask': True,  # if False, masking bond with 'BLANK'; if True, masking bond with 'MASK'

        'data_loader_check_valid': False,
        'shuffle_input': False,
    },
    'loss': {
        'cl_T': 0.07,
        'n_cl_pool': 0,
    },
    'loss_weight': {
        'l_single_bond_loss': 1.0,

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
})





















