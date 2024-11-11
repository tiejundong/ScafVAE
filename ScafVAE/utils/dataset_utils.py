import copy
import os

import random
from collections import defaultdict
import pickle
from functools import partial

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import MACCSkeys
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from torch.utils.data import DataLoader, Dataset
import torch_geometric

from ScafVAE.model.config import *
from ScafVAE.utils.common import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.graph2mol import atom_types, bond_types, get_state, quick_check



class PPLDataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_list, args):
        self.som_path = args.som_path
        self.mode = mode
        self.data_list = data_list
        self.args = args
        self.init_batch(args, mode)

        self.data_config = training_config.data
        self.loss_config = training_config.loss

        self.max_l_len = global_config.max_l_len
        self.n_dummy_atom = global_config.max_l_len
        self.mask_rate = self.data_config.mask_rate
        self.data_loader_check_valid = self.data_config.data_loader_check_valid

    def init_batch(self, args, mode):
        if args.n_batch == -1:
            self.n_batch = len(self.data_list)
        else:
            self.n_batch = min(args.n_batch, len(self.data_list))

        if mode == 'val' and self.n_batch > 0:
            self.n_batch = self.n_batch // 10

        if self.args.n_batch > 0:
            assert self.args.persistent_workers == False

        self.set_data_epoch(0)

    def set_data_epoch(self, epoch=0):
        # set persistent_workers to False
        start = self.n_batch * epoch % len(self.data_list)
        ext_data_list = self.data_list + self.data_list
        self.sub_data_list = ext_data_list[start: start + self.n_batch]

        if self.args.is_main_proc and self.args.n_batch > 0:
            print(f'Set dataset to [{start}:{start + self.n_batch}]')

    def __getitem__(self, idx):
        l_idx = self.sub_data_list[idx]
        dic_ligand = self.getitem_ligand(l_idx)
        return dic_ligand

    def getitem_ligand(self, l_idx):
        # get processed data
        dic_ligand = pickle.load(open(f'{self.som_path}/{l_idx}.pkl', 'rb'))
        dic_ligand['idx'] = l_idx

        # prepare ligand
        dic_ligand = self.get_ligand(dic_ligand)
        dic_ligand['max_l_len'] = self.max_l_len
        dic_ligand.update(self.prepare_ligand(dic_ligand))

        return dic_ligand

    def prepare_ligand(self, dic_ligand):
        # ========== add dummy atoms ==========
        if self.data_config.shuffle_input:
            perm_idx = np.arange(dic_ligand['l_x_init'].shape[0])
            np.random.shuffle(perm_idx)
            dic_ligand['l_x_init'] = dic_ligand['l_x_init'][perm_idx]
            dic_ligand['l_edge_init'] = dic_ligand['l_edge_init'][perm_idx, :]
            dic_ligand['l_edge_init'] = dic_ligand['l_edge_init'][:, perm_idx]

        # Warning: need same as in dataloader and data preprocess
        # ['UNK', 'C', 'O', 'N', ...] -> ['dummy atoms', 'UNK', 'C', 'O', 'N', ...]
        len_ligand = dic_ligand['len_ligand']
        assert len_ligand <= self.n_dummy_atom
        l_x_init = F.pad(dic_ligand['l_x_init'], (1, 0, 0, self.n_dummy_atom - len_ligand), 'constant', 0)
        l_x_init[len_ligand:, 0] = 1  # dummy ele
        l_x_init[len_ligand:, -len(Chem.rdchem.ChiralType.values)] = 1  # dummy ChiralType

        l_edge_init = F.pad(
            dic_ligand['l_edge_init'],
            (0, 0, 0, self.n_dummy_atom - len_ligand, 0, self.n_dummy_atom - len_ligand), 'constant', 0
        )
        l_edge_init[len_ligand:, :, 0] = 1
        l_edge_init[:, len_ligand:, 0] = 1

        # ========== label prediction ==========
        # ['dummy atoms', 'UNK', 'C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'B', 'I', 'P']
        l_x_label = l_x_init[:, :len(atom_types)].argmax(dim=-1)
        assert torch.all(l_x_init[:, :len(atom_types)].sum(dim=-1) == 1)

        # [Chem.rdchem.ChiralType]
        l_chiral_type = l_x_init[:, -len(Chem.rdchem.ChiralType.values):].argmax(dim=-1)
        assert torch.all(l_x_init[:, -len(Chem.rdchem.ChiralType.values):].sum(dim=-1) == 1)

        # ['BLANK', SINGLE, DOUBLE, TRIPLE, AROMATIC]
        l_edge_label = l_edge_init[..., :len(bond_types)].argmax(dim=-1)
        assert torch.all(l_edge_init[..., :len(bond_types)].sum(dim=-1) == 1)
        ele_list = [atom_types[i.item()] for i in l_x_label]
        l_bond_in_ring = l_edge_init[..., -1]

        # l_x_init = torch.concat([
        #     l_x_init[:, :len(atom_types)], l_x_init[:, -len(Chem.rdchem.ChiralType.values):]
        # ], dim=-1)
        l_x_init = l_x_init[:, :len(atom_types)]  # no ChiralType
        l_edge_init = l_edge_init[..., :len(bond_types)]

        # ========== valid check ==========
        if self.data_loader_check_valid:
            mol_state = get_state(l_x_label, l_edge_label)
            is_valid = quick_check(mol_state, return_with_obj=False)
            valid_mask = 1 if is_valid else 0
        else:
            valid_mask = 0

        # ========== masking ==========
        l_x_masked_init = F.pad(l_x_init, (0, 1), 'constant', 0)
        l_edge_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0)

        # for node
        atom_exist_mask = (l_x_label != 0).float()  # no dummy
        n_mask_node = max(2, int(self.mask_rate * atom_exist_mask.sum()))
        rand_mask_node_idx = torch.nonzero(atom_exist_mask).reshape(-1)[
            torch.randperm(atom_exist_mask.sum().int())[:n_mask_node]]
        rand_mask_node_idx_1 = rand_mask_node_idx[:n_mask_node // 2]
        rand_mask_node_idx_2 = rand_mask_node_idx[n_mask_node // 2:]
        l_x_init_masked_1 = torch.empty_like(l_x_masked_init).copy_(l_x_masked_init)
        l_x_init_masked_1[rand_mask_node_idx_1] = 0
        l_x_init_masked_1[rand_mask_node_idx_1, -1] = 1
        l_x_init_masked_2 = torch.empty_like(l_x_masked_init).copy_(l_x_masked_init)
        l_x_init_masked_2[rand_mask_node_idx_2] = 0
        l_x_init_masked_2[rand_mask_node_idx_2, -1] = 1

        # sym mask for contrastive_loss
        # To determine if two atoms are symmetric, the hash codes from the Morgan algorithm at a given radius are used
        # CR = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        # If breakTies is False, this returns the symmetry class for each atom.
        # The symmetry class is used by the canonicalization routines to type each atom based on the whole chemistry of
        # the molecular graph.Any atom with the same rank (symmetry class ) is indistinguishable.
        # x_contrastive_sym_mask = torch.zeros(l_len, l_len)
        # for i in range(l_len_no_dummy - 1):
        #     for j in range(i + 1, l_len_no_dummy):
        #         if CR[i] == CR[j]:
        #             x_contrastive_sym_mask[i, j] = 1
        #             x_contrastive_sym_mask[j, i] = 1

        # for edge
        bond_exist_mask = (l_edge_label != 0).float().triu(diagonal=1)  # no BLANK
        n_mask_edge = max(2, int(self.mask_rate * bond_exist_mask.sum()))
        rand_mask_edge_idx = torch.nonzero(bond_exist_mask)[
            torch.randperm(bond_exist_mask.sum().int())[:n_mask_edge]]
        rand_mask_edge_idx_1 = rand_mask_edge_idx[:n_mask_edge // 2]
        rand_mask_edge_idx_2 = rand_mask_edge_idx[n_mask_edge // 2:]
        rand_mask_edge_idx_1 = torch.concat([rand_mask_edge_idx_1, rand_mask_edge_idx_1.flip([-1])], dim=0).T
        rand_mask_edge_idx_2 = torch.concat([rand_mask_edge_idx_2, rand_mask_edge_idx_2.flip([-1])], dim=0).T
        l_edge_init_masked_1 = torch.empty_like(l_edge_masked_init).copy_(l_edge_masked_init)
        l_edge_init_masked_1[rand_mask_edge_idx_1[0], rand_mask_edge_idx_1[1]] = 0
        l_edge_init_masked_1[rand_mask_edge_idx_1[0], rand_mask_edge_idx_1[1], -1] = 1
        l_edge_init_masked_2 = torch.empty_like(l_edge_masked_init).copy_(l_edge_masked_init)
        l_edge_init_masked_2[rand_mask_edge_idx_2[0], rand_mask_edge_idx_2[1]] = 0
        l_edge_init_masked_2[rand_mask_edge_idx_2[0], rand_mask_edge_idx_2[1], -1] = 1

        # ========== single bond masking ==========
        bond_exist_mask = (l_edge_label != 0).float().triu(diagonal=1)  # no BLANK
        if torch.rand(1) < 0.5:  # random choose bond / non-bond
            bond_exist_mask = (1 - bond_exist_mask).triu(diagonal=1)
            bond_exist_mask[len_ligand:, :] = 0
            bond_exist_mask[:, len_ligand:] = 0
        if bond_exist_mask.sum() == 0:
            l_edge_single_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0)
            mask_for_edge_single_masked = torch.zeros(l_edge_init.shape[:-1])
        else:
            rand_mask_edge_idx = torch.nonzero(bond_exist_mask)[torch.randperm(bond_exist_mask.sum().int())[0]]
            rand_mask_edge_idx = torch.stack([rand_mask_edge_idx, rand_mask_edge_idx.flip([-1])], dim=0)
            l_edge_single_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0)
            l_edge_single_masked_init[rand_mask_edge_idx[0], rand_mask_edge_idx[1]] = 0
            l_edge_single_masked_init[rand_mask_edge_idx[0], rand_mask_edge_idx[1], -1] = 1
            mask_for_edge_single_masked = torch.zeros(l_edge_init.shape[:-1])
            mask_for_edge_single_masked[rand_mask_edge_idx[0], rand_mask_edge_idx[1]] = 1

        # ========== single bond masking for all bonds ==========
        bond_exist_mask = (l_edge_label != 0).float().triu(diagonal=1)  # no BLANK
        if bond_exist_mask.sum() == 0:
            l_edge_all_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0).unsqueeze(dim=0)
            mask_for_edge_all_masked = torch.zeros(l_edge_init.shape[:-1]).unsqueeze(dim=0)
        else:
            rand_mask_edge_idx = torch.nonzero(bond_exist_mask).T
            l_edge_all_masked_init = []
            mask_for_edge_all_masked = []
            for i in range(rand_mask_edge_idx.shape[-1]):
                init = F.pad(l_edge_init, (0, 1), 'constant', 0)
                init[rand_mask_edge_idx[0, i], rand_mask_edge_idx[1, i]] = 0
                init[rand_mask_edge_idx[0, i], rand_mask_edge_idx[1, i], -1] = 1
                init[rand_mask_edge_idx[1, i], rand_mask_edge_idx[0, i]] = 0
                init[rand_mask_edge_idx[1, i], rand_mask_edge_idx[0, i], -1] = 1
                mask = torch.zeros(l_edge_init.shape[:-1])
                mask[rand_mask_edge_idx[0, i], rand_mask_edge_idx[1, i]] = 1
                mask[rand_mask_edge_idx[1, i], rand_mask_edge_idx[0, i]] = 1
                l_edge_all_masked_init.append(init)
                mask_for_edge_all_masked.append(mask)
            l_edge_all_masked_init = torch.stack(l_edge_all_masked_init, dim=0)
            mask_for_edge_all_masked = torch.stack(mask_for_edge_all_masked, dim=0)

        # ========== output ==========
        dic_input = dict(
            # ========== input ==========
            l_x_init=l_x_init,
            l_edge_init=l_edge_init,

            # ========== supply info ==========
            l_x_label=l_x_label.long(),
            l_chiral_type=l_chiral_type.long(),
            l_edge_label=l_edge_label.long(),
            l_bond_in_ring=l_bond_in_ring,

            l_x_init_masked_1=l_x_init_masked_1,
            l_x_init_masked_2=l_x_init_masked_2,
            l_edge_init_masked_1=l_edge_init_masked_1,
            l_edge_init_masked_2=l_edge_init_masked_2,

            l_edge_single_masked_init=l_edge_single_masked_init,
            mask_for_edge_single_masked=mask_for_edge_single_masked,
            l_edge_all_masked_init=l_edge_all_masked_init,
            mask_for_edge_all_masked=mask_for_edge_all_masked,

            ele_list=ele_list,
            valid_mask=valid_mask,
        )

        return dic_input

    def get_ligand(self, dic_ligand):
        len_ligand = len(dic_ligand['ligand_node_features'])
        l_match = dic_ligand['ligand_match'].reshape(-1)
        n_match = len(l_match) // len_ligand
        l_nomatch = repeat(np.arange(0, len_ligand), 'm -> (n m)', n=n_match)

        l_x_init = dic_ligand['ligand_node_features']
        l_edge_init = dic_ligand['ligand_edge_features']

        # distance map
        l_dismap_mask = np.where(dic_ligand['ligand_dismap'] == -1, 0, 1)

        # aff
        if 'aff' not in dic_ligand.keys():
            dic_ligand['aff'] = -1

        dic_ligand = dict(
            l_x_init=torch.from_numpy(l_x_init).float(),
            l_edge_init=torch.from_numpy(l_edge_init).float(),
            l_coor_true=torch.from_numpy(dic_ligand['ligand_coor_true']).float(),
            l_dismap=torch.from_numpy(dic_ligand['ligand_dismap']).float(),
            l_dismap_mask=torch.from_numpy(l_dismap_mask).float(),
            l_aff=torch.Tensor([dic_ligand['aff']]).float(),

            len_ligand=len_ligand,
            l_match=torch.from_numpy(l_match).long(),
            l_nomatch=torch.from_numpy(l_nomatch).long(),

            # ligand_CR=dic_ligand['ligand_CR'],
            # l_ref_coor=torch.from_numpy(dic_ligand['l_ref_coor']).float(),
            idx=dic_ligand['idx'],
        )

        return dic_ligand

    def __len__(self):
        return self.n_batch


def collate_ppl(batch_list):
    if 'max_l_len' in batch_list[0].keys() and batch_list[0]['max_l_len'] is not None:
        max_l_len = batch_list[0]['max_l_len']
    else:
        max_l_len = 0
    for g in batch_list:
        g['l_mask'] = torch.ones(g['len_ligand'])
        max_l_len = max(max_l_len, g['len_ligand'])

    # ========== feat, coor and mask ==========
    dic_ligand = {}
    dic_ligand.update(pad_zeros(batch_list, [
        'l_x_init', 'l_coor_true', 'l_ref_coor',
        'l_x_init_masked_1', 'l_x_init_masked_2',
    ], max_l_len, collect_dim=-3, data_type='1d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_mask',
        'l_x_label', 'l_chiral_type',
    ], max_l_len, collect_dim=-2, data_type='1d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_edge_init',
        'l_edge_init_masked_1', 'l_edge_init_masked_2',
        'l_edge_single_masked_init',
    ], max_l_len, collect_dim=-4, data_type='2d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_edge_label', 'l_bond_in_ring',
        'mask_for_edge_single_masked', 'l_dismap_mask',
    ], max_l_len, collect_dim=-3, data_type='2d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_dismap',
    ], max_l_len, collect_dim=-3, data_type='2d', value=-1))


    # ========== mask for all bonds ==========
    if 'l_edge_all_masked_init' in batch_list[0].keys():
        dic_ligand['l_edge_all_masked_init'] = torch.concat(
            [F.pad(
                g['l_edge_all_masked_init'],
                (0, 0,
                 0, max_l_len - g['l_edge_all_masked_init'].shape[-2],
                 0, max_l_len - g['l_edge_all_masked_init'].shape[-2]),
                'constant', 0) for g in batch_list], dim=0,
        )
        dic_ligand['mask_for_edge_all_masked'] = torch.concat(
            [F.pad(
                g['mask_for_edge_all_masked'],
                (0, max_l_len - g['mask_for_edge_all_masked'].shape[-2],
                 0, max_l_len - g['mask_for_edge_all_masked'].shape[-2]),
                'constant', 0) for g in batch_list], dim=0,
        )
        scatter_idx_for_edge_all_masked = []
        for i, g in enumerate(batch_list):
            scatter_idx_for_edge_all_masked += [i] * g['l_edge_all_masked_init'].shape[0]
        dic_ligand['scatter_idx_for_edge_all_masked'] = torch.LongTensor(scatter_idx_for_edge_all_masked)
    else:
        dic_ligand['l_edge_all_masked_init'] = torch.zeros(1)
        dic_ligand['mask_for_edge_all_masked'] = torch.zeros(1)
        dic_ligand['scatter_idx_for_edge_all_masked'] = torch.zeros(1)


    # ========== suppl ==========
    dic_ligand['idx'] = [g['idx'] for g in batch_list]
    dic_ligand['l_aff'] = torch.concat([g['l_aff'] for g in batch_list])
    dic_ligand['len_ligand'] = torch.LongTensor([g['len_ligand'] for g in batch_list])
    dic_ligand['valid_mask'] = torch.Tensor([g['valid_mask'] for g in batch_list])
    dic_ligand['ele_list'] = [g['ele_list'] for g in batch_list]


    # ========== ligand matching ==========
    len_tmp_ligand = 0
    len_tmp_batch_match = 0
    l_loc = []
    l_match = []
    l_nomatch = []
    scatter_ligand_1 = []
    scatter_ligand_2 = []
    scatter_ligand_3 = []
    for i, g in enumerate(batch_list):
        l_loc.append(torch.arange(g['len_ligand']) + i * max_l_len)
        l_match.append(g['l_match'] + len_tmp_ligand)
        l_nomatch.append(g['l_nomatch'] + len_tmp_ligand)
        scatter_ligand_1.append(repeat(torch.arange(0, len(g['l_match']) // g['len_ligand']),
                                       'i -> (i m)', m=g['len_ligand']) + len_tmp_batch_match)
        scatter_ligand_2.append(torch.zeros(len(g['l_match']) // g['len_ligand']) + i)
        scatter_ligand_3.append(torch.zeros(g['len_ligand']) + i)
        len_tmp_ligand += g['len_ligand']
        len_tmp_batch_match += len(g['l_match']) // g['len_ligand']
    dic_ligand['l_loc'] = torch.cat(l_loc, dim=0).long()
    dic_ligand['l_match'] = torch.cat(l_match, dim=0).long()
    dic_ligand['l_nomatch'] = torch.cat(l_nomatch, dim=0).long()
    dic_ligand['scatter_ligand_1'] = torch.concat(scatter_ligand_1, dim=0).long()
    dic_ligand['scatter_ligand_2'] = torch.concat(scatter_ligand_2, dim=0).long()
    dic_ligand['scatter_ligand_3'] = torch.concat(scatter_ligand_3, dim=0).long()

    return dic_ligand



class ComplexDataset(Dataset):
    def __init__(self, mode, data_list, args):
        self.data_path = args.data_path
        self.extra_path = args.extra_path.split(',')
        self.task_name = args.config
        self.mode = mode
        self.data_list = data_list
        self.cache_path = args.cache_path
        self.args = args
        self.init_batch(args, mode)

        self.model_config = ScafVAE_config
        self.data_config = training_config.data
        self.loss_config = training_config.loss

        if mode == 'train':
            self.max_l_len = global_config.max_l_len
            self.n_dummy_atom = global_config.max_l_len
        else:
            self.max_l_len = global_config.max_l_len_for_eval
            self.n_dummy_atom = global_config.max_l_len_for_eval
        self.mask_rate = self.data_config.mask_rate
        self.data_loader_check_valid = self.data_config.data_loader_check_valid

        self.dic_scaf_idx = {k: idx for idx, k in enumerate(pd.read_csv(frag_codebook_path)['scaf_smi'].values)}

    def init_batch(self, args, mode):
        if args.n_batch == -1:
            self.n_batch = len(self.data_list)
        else:
            self.n_batch = min(args.n_batch, len(self.data_list))

        if mode == 'val' and self.n_batch > 0:
            self.n_batch = self.n_batch // 20
            
        if self.args.n_batch > 0:
            assert self.args.persistent_workers == False
           
        self.set_data_epoch(0)

    def set_data_epoch(self, epoch=0):
        # set persistent_workers to False
        start = self.n_batch * epoch % len(self.data_list)
        ext_data_list = self.data_list + self.data_list
        self.sub_data_list = ext_data_list[start: start + self.n_batch]

        if self.args.is_main_proc and self.args.n_batch > 0:
            print(f'Set dataset to [{start}:{start + self.n_batch}]')

    def __getitem__(self, idx):
        l_idx = self.sub_data_list[idx]

        # control scaf num
        # if self.data_config.n_max_scaf is not None:  # and self.mode == 'train':
        #     assert self.task_name == 'FRAG'
        #     n_loop_count = 0
        #     while True:
        #         n_loop_count += 1
        #         assert n_loop_count < 1000
        #         dic_scaf = np.load(f'{self.extra_path[0]}/{l_idx}.npz')
        #         n = len(str(dic_scaf['frag_smi_combined']).split(','))
        #         if n <= self.data_config.n_max_scaf:
        #             break
        #         else:
        #             l_idx = random.choice(self.sub_data_list)

        # get base feature
        dic_ligand = self.getitem_ligand(l_idx)

        # get scaf relative feature
        if self.task_name == 'FRAG':
            dic_ligand.update(self.get_scaf(l_idx))

        # reduce batch size
        if self.task_name == 'FRAG':
            dic_ligand['max_drop_size'] = self.data_config.max_drop_size
            dic_ligand['max_drop_retain'] = self.data_config.max_drop_retain
        else:
            dic_ligand['max_drop_size'] = None
            dic_ligand['max_drop_retain'] = None
        dic_ligand['mode'] = self.mode

        return dic_ligand

    def get_scaf(self, idx):
        dic_scaf = np.load(f'{self.extra_path[0]}/{idx}.npz')

        # zero for dummy and pad positions
        frag_smi = str(dic_scaf['frag_smi_combined']).split(',')
        N_frag = len(frag_smi)
        scaf_idx = torch.LongTensor([self.dic_scaf_idx[k] for k in frag_smi])  # reordered scaf_idx
        scaf_idx = scaf_idx + 1  # use 0 for dummy atom pooling i.e. pad_token

        reordered_component_idx = torch.from_numpy(dic_scaf['reordered_component_idx']).long()
        reordered_component_idx = F.pad(
            reordered_component_idx,
            (0, self.max_l_len-len(reordered_component_idx)),
            'constant', N_frag)
        reordered_edge = torch.from_numpy(dic_scaf['reordered_edge']).float()
        N = reordered_edge.shape[0]
        reordered_edge = F.pad(
            reordered_edge,
            (0, 0, 0, self.max_l_len-N, 0, self.max_l_len-N),
            'constant', 0)
        reordered_edge[:, N:, 0] = 1
        reordered_edge[N:, :, 0] = 1

        # get sparse frag adj
        reordered_scaf_sparse_adj = get_sparse_scaf(
            reordered_edge, reordered_component_idx, self.model_config.ppl_force_max_atom_with_ring).float()

        # bfs
        bfs_idx = bfs_sort(
            scaf_idx,
            reordered_scaf_sparse_adj,
            blank_node=0,
            blank_edge=self.model_config.ppl_force_max_atom_with_ring ** 2,
        )
        bfs_idx_reverse = bfs_idx.argsort(dim=-1, descending=False)

        scaf_idx_bfs = scaf_idx[bfs_idx]
        reordered_scaf_sparse_adj_bfs = reordered_scaf_sparse_adj[bfs_idx, :][:, bfs_idx]

        scaf_idx_bfs_with_start_and_end = F.pad(
            scaf_idx_bfs, (1, 0), 'constant', self.model_config.x_start_token)
        scaf_idx_bfs_with_start_and_end = F.pad(
            scaf_idx_bfs_with_start_and_end, (0, 1), 'constant', self.model_config.x_end_token)

        component_idx = torch.from_numpy(dic_scaf['component_idx']).long()
        if len(component_idx) < self.max_l_len:
            component_idx = F.pad(
                component_idx,
                (0, self.max_l_len - len(component_idx)),
                'constant',
                N_frag,
            )

        dic_output = dict(
            component_idx=component_idx,

            reordered_component_idx=reordered_component_idx,
            reordered_edge=reordered_edge,
            scaf_idx=scaf_idx.long(),
            N_frag=N_frag,
            reordered_scaf_sparse_adj=reordered_scaf_sparse_adj.long(),

            bfs_idx=bfs_idx.long(),
            bfs_idx_reverse=bfs_idx_reverse.long(),
            scaf_idx_bfs=scaf_idx_bfs.long(),
            reordered_scaf_sparse_adj_bfs=reordered_scaf_sparse_adj_bfs.long(),
            scaf_idx_bfs_with_start_and_end=scaf_idx_bfs_with_start_and_end.long(),
        )

        del dic_scaf

        return dic_output

    def getitem_ligand(self, l_idx):
        # get processed data
        dic_ligand_np = pickle.load(open(f'{self.data_path}/{l_idx}.pkl', 'rb'))
        dic_ligand_np['idx'] = l_idx

        # prepare ligand
        dic_ligand = self.get_ligand(dic_ligand_np)
        dic_ligand['max_l_len'] = self.max_l_len
        dic_ligand.update(self.prepare_ligand(dic_ligand))

        return dic_ligand

    def prepare_ligand(self, dic_ligand):
        # ========== add dummy atoms ==========
        if self.data_config.shuffle_input:
            perm_idx = np.arange(dic_ligand['l_x_init'].shape[0])
            np.random.shuffle(perm_idx)
            dic_ligand['l_x_init'] = dic_ligand['l_x_init'][perm_idx]
            dic_ligand['l_edge_init'] = dic_ligand['l_edge_init'][perm_idx, :]
            dic_ligand['l_edge_init'] = dic_ligand['l_edge_init'][:, perm_idx]

        # Warning: need same as in dataloader and data preprocess
        # ['UNK', 'C', 'O', 'N', ...] -> ['dummy atoms', 'UNK', 'C', 'O', 'N', ...]
        len_ligand = dic_ligand['len_ligand']
        assert len_ligand <= self.n_dummy_atom
        l_x_init = F.pad(dic_ligand['l_x_init'], (1, 0, 0, self.n_dummy_atom - len_ligand), 'constant', 0)
        l_x_init[len_ligand:, 0] = 1  # dummy ele
        l_x_init[len_ligand:, -len(Chem.rdchem.ChiralType.values)] = 1  # dummy ChiralType

        l_edge_init = F.pad(
            dic_ligand['l_edge_init'],
            (0, 0, 0, self.n_dummy_atom - len_ligand, 0, self.n_dummy_atom - len_ligand), 'constant', 0
        )
        l_edge_init[len_ligand:, :, 0] = 1
        l_edge_init[:, len_ligand:, 0] = 1


        # ========== label prediction ==========
        # ['dummy atoms', 'UNK', 'C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'B', 'I', 'P']
        l_x_label = l_x_init[:, :len(atom_types)].argmax(dim=-1)
        assert torch.all(l_x_init[:, :len(atom_types)].sum(dim=-1) == 1)

        # [Chem.rdchem.ChiralType]
        l_chiral_type = l_x_init[:, -len(Chem.rdchem.ChiralType.values):].argmax(dim=-1)
        assert torch.all(l_x_init[:, -len(Chem.rdchem.ChiralType.values):].sum(dim=-1) == 1)

        # ['BLANK', SINGLE, DOUBLE, TRIPLE, AROMATIC]
        l_edge_label = l_edge_init[..., :len(bond_types)].argmax(dim=-1)
        assert torch.all(l_edge_init[..., :len(bond_types)].sum(dim=-1) == 1)
        ele_list = [atom_types[i.item()] for i in l_x_label]
        l_bond_in_ring = l_edge_init[..., -1]

        # l_x_init = torch.concat([
        #     l_x_init[:, :len(atom_types)], l_x_init[:, -len(Chem.rdchem.ChiralType.values):]
        # ], dim=-1)
        l_x_init = l_x_init[:, :len(atom_types)]  # no ChiralType
        l_edge_init = l_edge_init[..., :len(bond_types)]


        # ========== valid check ==========
        if self.data_loader_check_valid:
            mol_state = get_state(l_x_label, l_edge_label)
            is_valid = quick_check(mol_state, return_with_obj=False)
            valid_mask = 1 if is_valid else 0
        else:
            valid_mask = 0


        # ========== masking ==========
        l_x_masked_init = F.pad(l_x_init, (0, 1), 'constant', 0)
        l_edge_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0)

        # for node
        atom_exist_mask = (l_x_label != 0).float()  # no dummy
        n_mask_node = max(2, int(self.mask_rate * 2 * atom_exist_mask.sum()))
        rand_mask_node_idx = torch.nonzero(atom_exist_mask).reshape(-1)[
            torch.randperm(atom_exist_mask.sum().int())[:n_mask_node]]
        rand_mask_node_idx_1 = rand_mask_node_idx[:n_mask_node // 2]
        rand_mask_node_idx_2 = rand_mask_node_idx[n_mask_node // 2:]
        l_x_init_masked_1 = torch.empty_like(l_x_masked_init).copy_(l_x_masked_init)
        l_x_init_masked_1[rand_mask_node_idx_1] = 0
        l_x_init_masked_1[rand_mask_node_idx_1, -1] = 1
        l_x_init_masked_2 = torch.empty_like(l_x_masked_init).copy_(l_x_masked_init)
        l_x_init_masked_2[rand_mask_node_idx_2] = 0
        l_x_init_masked_2[rand_mask_node_idx_2, -1] = 1

        # sym mask for contrastive_loss
        # To determine if two atoms are symmetric, the hash codes from the Morgan algorithm at a given radius are used
        # CR = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        # If breakTies is False, this returns the symmetry class for each atom.
        # The symmetry class is used by the canonicalization routines to type each atom based on the whole chemistry of
        # the molecular graph.Any atom with the same rank (symmetry class ) is indistinguishable.
        # x_contrastive_sym_mask = torch.zeros(l_len, l_len)
        # for i in range(l_len_no_dummy - 1):
        #     for j in range(i + 1, l_len_no_dummy):
        #         if CR[i] == CR[j]:
        #             x_contrastive_sym_mask[i, j] = 1
        #             x_contrastive_sym_mask[j, i] = 1

        # for edge
        bond_exist_mask = (l_edge_label != 0).float().triu(diagonal=1)  # no BLANK
        n_mask_edge = max(2, int(self.mask_rate * 2 * bond_exist_mask.sum()))
        rand_mask_edge_idx = torch.nonzero(bond_exist_mask)[
            torch.randperm(bond_exist_mask.sum().int())[:n_mask_edge]]
        rand_mask_edge_idx_1 = rand_mask_edge_idx[:n_mask_edge // 2]
        rand_mask_edge_idx_2 = rand_mask_edge_idx[n_mask_edge // 2:]
        rand_mask_edge_idx_1 = torch.concat([rand_mask_edge_idx_1, rand_mask_edge_idx_1.flip([-1])], dim=0).T
        rand_mask_edge_idx_2 = torch.concat([rand_mask_edge_idx_2, rand_mask_edge_idx_2.flip([-1])], dim=0).T
        l_edge_init_masked_1 = torch.empty_like(l_edge_masked_init).copy_(l_edge_masked_init)
        l_edge_init_masked_1[rand_mask_edge_idx_1[0], rand_mask_edge_idx_1[1]] = 0
        l_edge_init_masked_1[rand_mask_edge_idx_1[0], rand_mask_edge_idx_1[1], -1] = 1
        l_edge_init_masked_2 = torch.empty_like(l_edge_masked_init).copy_(l_edge_masked_init)
        l_edge_init_masked_2[rand_mask_edge_idx_2[0], rand_mask_edge_idx_2[1]] = 0
        l_edge_init_masked_2[rand_mask_edge_idx_2[0], rand_mask_edge_idx_2[1], -1] = 1

        # ========== single bond masking ==========
        bond_exist_mask = (l_edge_label != 0).float().triu(diagonal=1)  # no BLANK
        if torch.rand(1) < 0.5:  # random choose bond / non-bond
            bond_exist_mask = (1 - bond_exist_mask).triu(diagonal=1)
            bond_exist_mask[len_ligand:, :] = 0
            bond_exist_mask[:, len_ligand:] = 0
        if bond_exist_mask.sum() == 0:
            l_edge_single_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0)
            mask_for_edge_single_masked = torch.zeros(l_edge_init.shape[:-1])
        else:
            rand_mask_edge_idx = torch.nonzero(bond_exist_mask)[torch.randperm(bond_exist_mask.sum().int())[0]]
            rand_mask_edge_idx = torch.stack([rand_mask_edge_idx, rand_mask_edge_idx.flip([-1])], dim=0)
            l_edge_single_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0)
            l_edge_single_masked_init[rand_mask_edge_idx[0], rand_mask_edge_idx[1]] = 0
            l_edge_single_masked_init[rand_mask_edge_idx[0], rand_mask_edge_idx[1], -1] = 1
            mask_for_edge_single_masked = torch.zeros(l_edge_init.shape[:-1])
            mask_for_edge_single_masked[rand_mask_edge_idx[0], rand_mask_edge_idx[1]] = 1

        # ========== single bond masking for all bonds ==========
        bond_exist_mask = (l_edge_label != 0).float().triu(diagonal=1)  # no BLANK
        if bond_exist_mask.sum() == 0:
            l_edge_all_masked_init = F.pad(l_edge_init, (0, 1), 'constant', 0).unsqueeze(dim=0)
            mask_for_edge_all_masked = torch.zeros(l_edge_init.shape[:-1]).unsqueeze(dim=0)
        else:
            rand_mask_edge_idx = torch.nonzero(bond_exist_mask).T
            l_edge_all_masked_init = []
            mask_for_edge_all_masked = []
            for i in range(rand_mask_edge_idx.shape[-1]):
                init = F.pad(l_edge_init, (0, 1), 'constant', 0)
                init[rand_mask_edge_idx[0, i], rand_mask_edge_idx[1, i]] = 0
                init[rand_mask_edge_idx[0, i], rand_mask_edge_idx[1, i], -1] = 1
                init[rand_mask_edge_idx[1, i], rand_mask_edge_idx[0, i]] = 0
                init[rand_mask_edge_idx[1, i], rand_mask_edge_idx[0, i], -1] = 1
                mask = torch.zeros(l_edge_init.shape[:-1])
                mask[rand_mask_edge_idx[0, i], rand_mask_edge_idx[1, i]] = 1
                mask[rand_mask_edge_idx[1, i], rand_mask_edge_idx[0, i]] = 1
                l_edge_all_masked_init.append(init)
                mask_for_edge_all_masked.append(mask)
            l_edge_all_masked_init = torch.stack(l_edge_all_masked_init, dim=0)
            mask_for_edge_all_masked = torch.stack(mask_for_edge_all_masked, dim=0)

        # ========== output ==========
        dic_input = dict(
            # ========== input ==========
            l_x_init=l_x_init,
            l_edge_init=l_edge_init,

            # ========== supply info ==========
            l_x_label=l_x_label.long(),
            l_chiral_type=l_chiral_type.long(),
            l_edge_label=l_edge_label.long(),
            l_bond_in_ring=l_bond_in_ring,

            l_x_init_masked_1=l_x_init_masked_1,
            l_x_init_masked_2=l_x_init_masked_2,
            l_edge_init_masked_1=l_edge_init_masked_1,
            l_edge_init_masked_2=l_edge_init_masked_2,

            l_edge_single_masked_init=l_edge_single_masked_init,
            mask_for_edge_single_masked=mask_for_edge_single_masked,
            l_edge_all_masked_init=l_edge_all_masked_init,
            mask_for_edge_all_masked=mask_for_edge_all_masked,

            ele_list=ele_list,
            valid_mask=valid_mask,
        )

        return dic_input

    def get_ligand(self, dic_ligand):
        len_ligand = len(dic_ligand['ligand_node_features'])
        l_match = dic_ligand['ligand_match'].reshape(-1)
        n_match = len(l_match) // len_ligand
        l_nomatch = repeat(np.arange(0, len_ligand), 'm -> (n m)', n=n_match)

        l_x_init = dic_ligand['ligand_node_features']
        l_edge_init = dic_ligand['ligand_edge_features']

        # distance map
        l_dismap_mask = np.where(dic_ligand['ligand_dismap'] == -1, 0, 1)

        # fp
        mol = Chem.MolFromSmiles(dic_ligand['smi'])

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=Aux_config.fp_mogan_hidden)
        fp_arr_1 = np.zeros(Aux_config.fp_mogan_hidden)
        DataStructs.ConvertToNumpyArray(fp, fp_arr_1)

        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_arr_2 = np.zeros(167)
        DataStructs.ConvertToNumpyArray(fp, fp_arr_2)

        fp_arr = np.concatenate([fp_arr_1, fp_arr_2])

        # smi
        smi = dic_ligand['smi']
        # smi_with_start_and_end = smi_start_token + smi + smi_end_token
        # smi_token_idx = np.array([dic_smi_encoder[s] for s in smi_with_start_and_end])

        # aff
        if 'aff' not in dic_ligand.keys():
            dic_ligand['aff'] = -1

        dic_output = dict(
            l_x_init=torch.from_numpy(l_x_init).float(),
            l_edge_init=torch.from_numpy(l_edge_init).float(),
            l_coor_true=torch.from_numpy(dic_ligand['ligand_coor_true']).float(),
            l_dismap=torch.from_numpy(dic_ligand['ligand_dismap']).float(),
            l_dismap_mask=torch.from_numpy(l_dismap_mask).float(),
            l_aff=torch.Tensor([dic_ligand['aff']]).float(),

            fp=torch.from_numpy(fp_arr).float(),

            len_ligand=len_ligand,
            l_match=torch.from_numpy(l_match).long(),
            l_nomatch=torch.from_numpy(l_nomatch).long(),

            # ligand_CR=dic_ligand['ligand_CR'],
            # l_ref_coor=torch.from_numpy(dic_ligand['l_ref_coor']).float(),
            smi=smi,
            # smi_token_idx=torch.from_numpy(smi_token_idx).long(),
            idx=dic_ligand['idx'],
        )

        del dic_ligand

        return dic_output

    def __len__(self):
        return self.n_batch


def get_max_len(batch_list, k, dim):
    max_len = 0
    for g in batch_list:
        max_len = max(max_len, g[k].shape[dim])
    return max_len


def reduce_batch_size(batch_list):
    max_drop_size = batch_list[0]['max_drop_size']
    max_drop_retain = batch_list[0]['max_drop_retain']
    mode = batch_list[0]['mode']

    if max_drop_size is None or mode != 'train':
        return batch_list

    large_batch_list = []
    small_batch_list = []
    for g in batch_list:
        if g['N_frag'] > max_drop_size:
            large_batch_list.append(g)
        else:
            small_batch_list.append(g)

    if len(large_batch_list) == 0:
        return batch_list

    if len(large_batch_list) > max_drop_retain:
        random.shuffle(large_batch_list)
        large_batch_list = large_batch_list[:max_drop_retain]

    if np.random.rand(1) > 0.5:
        return large_batch_list
    else:
        return small_batch_list


def collate_ligand(batch_list):
    batch_list = reduce_batch_size(batch_list)

    if 'max_l_len' in batch_list[0].keys() and batch_list[0]['max_l_len'] is not None:
        max_l_len = batch_list[0]['max_l_len']
    else:
        max_l_len = 0
    for g in batch_list:
        g['l_mask'] = torch.ones(g['len_ligand'])
        max_l_len = max(max_l_len, g['len_ligand'])


    # ========== feat, coor and mask ==========
    dic_ligand = {}
    dic_ligand.update(pad_zeros(batch_list, [
        'l_x_init', 'l_coor_true', 'l_ref_coor',
        'l_x_init_masked_1', 'l_x_init_masked_2',
    ], max_l_len, collect_dim=-3, data_type='1d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_mask',
        'l_x_label', 'l_chiral_type',
    ], max_l_len, collect_dim=-2, data_type='1d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_edge_init',
        'l_edge_init_masked_1', 'l_edge_init_masked_2',
        'l_edge_single_masked_init',
    ], max_l_len, collect_dim=-4, data_type='2d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_edge_label', 'l_bond_in_ring',
        'mask_for_edge_single_masked', 'l_dismap_mask',
    ], max_l_len, collect_dim=-3, data_type='2d'))
    dic_ligand.update(pad_zeros(batch_list, [
        'l_dismap',
    ], max_l_len, collect_dim=-3, data_type='2d', value=-1))


    # ========== mask for all bonds ==========
    if 'l_edge_all_masked_init' in batch_list[0].keys():
        dic_ligand['l_edge_all_masked_init'] = torch.concat(
            [F.pad(
                g['l_edge_all_masked_init'],
                (0, 0,
                 0, max_l_len - g['l_edge_all_masked_init'].shape[-2],
                 0, max_l_len - g['l_edge_all_masked_init'].shape[-2]),
                'constant', 0) for g in batch_list], dim=0,
        )
        dic_ligand['mask_for_edge_all_masked'] = torch.concat(
            [F.pad(
                g['mask_for_edge_all_masked'],
                (0, max_l_len - g['mask_for_edge_all_masked'].shape[-2],
                 0, max_l_len - g['mask_for_edge_all_masked'].shape[-2]),
                'constant', 0) for g in batch_list], dim=0,
        )
        scatter_idx_for_edge_all_masked = []
        for i, g in enumerate(batch_list):
            scatter_idx_for_edge_all_masked += [i] * g['l_edge_all_masked_init'].shape[0]
        dic_ligand['scatter_idx_for_edge_all_masked'] = torch.LongTensor(scatter_idx_for_edge_all_masked)
    else:
        dic_ligand['l_edge_all_masked_init'] = torch.zeros(1)
        dic_ligand['mask_for_edge_all_masked'] = torch.zeros(1)
        dic_ligand['scatter_idx_for_edge_all_masked'] = torch.zeros(1)


    # ========== suppl ==========
    dic_ligand['idx'] = [g['idx'] for g in batch_list]
    dic_ligand['l_aff'] = torch.concat([g['l_aff'] for g in batch_list])
    dic_ligand['len_ligand'] = torch.LongTensor([g['len_ligand'] for g in batch_list])
    dic_ligand['valid_mask'] = torch.Tensor([g['valid_mask'] for g in batch_list])
    dic_ligand['ele_list'] = [g['ele_list'] for g in batch_list]
    dic_ligand['smi'] = [g['smi'] for g in batch_list]
    dic_ligand['fp'] = torch.stack([g['fp'] for g in batch_list])


    # ========== smi ==========
    # max_smi_len = get_max_len(batch_list, 'smi_token_idx', 0)
    # dic_ligand.update(pad_zeros(batch_list, [
    #     'smi_token_idx',
    # ], max_smi_len, collect_dim=-2, data_type='1d', value=smi_tokens.index(smi_pad_token)))


    # ========== scaf ==========
    if 'component_idx' in batch_list[0].keys():
        dic_ligand['component_idx'] = torch.stack([g['component_idx'] for g in batch_list], dim=0)

    if 'reordered_component_idx' in batch_list[0].keys():
        dic_ligand['reordered_component_idx'] = torch.stack([g['reordered_component_idx'] for g in batch_list], dim=0)

    if 'reordered_edge' in batch_list[0].keys():
        dic_ligand['reordered_edge'] = torch.stack([g['reordered_edge'] for g in batch_list], dim=0)

    if 'scaf_idx' in batch_list[0].keys():
        max_bfs_idx_len = get_max_len(batch_list, 'bfs_idx', 0)
        bfs_idx = []
        bfs_idx_reverse = []
        for g in batch_list:
            sub_bfs_idx = torch.arange(max_bfs_idx_len)
            sub_bfs_idx[:len(g['bfs_idx'])] = g['bfs_idx']
            bfs_idx.append(sub_bfs_idx)

            sub_bfs_idx_reverse = torch.arange(max_bfs_idx_len)
            sub_bfs_idx_reverse[:len(g['bfs_idx_reverse'])] = g['bfs_idx_reverse']
            bfs_idx_reverse.append(sub_bfs_idx_reverse)
        dic_ligand['bfs_idx'] = torch.stack(bfs_idx, dim=0)
        dic_ligand['bfs_idx_reverse'] = torch.stack(bfs_idx_reverse, dim=0)

        max_scaf_len = get_max_len(batch_list, 'scaf_idx', 0)
        dic_ligand.update(pad_zeros(batch_list, [
            'scaf_idx', 'scaf_idx_bfs',
        ], max_scaf_len, collect_dim=-2, data_type='1d', value=0))

        # max_len = get_max_len(batch_list, 'reordered_scaf_sparse_adj', 0)
        dic_ligand.update(pad_zeros(batch_list, [
            'reordered_scaf_sparse_adj', 'reordered_scaf_sparse_adj_bfs',
        ], max_scaf_len, collect_dim=-3, data_type='2d', value=global_config.ppl_force_max_atom_with_ring**2))

        max_scaf_with_start_and_end_len = get_max_len(batch_list, 'scaf_idx_bfs_with_start_and_end', 0)
        dic_ligand.update(pad_zeros(batch_list, [
            'scaf_idx_bfs_with_start_and_end',
        ], max_scaf_with_start_and_end_len, collect_dim=-2, data_type='1d', value=0))

    return dic_ligand



class FinetuningDataset(Dataset):
    def __init__(self, mode, data_list, args, y_list):
        self.base_dataset = ComplexDataset(mode, data_list, args)
        self.base_dataset.n_batch = len(data_list)
        self.base_dataset.sub_data_list = data_list

        self.data_list = data_list
        self.dic_y_list = {idx: y for idx, y in zip(data_list, y_list)}

    def __getitem__(self, idx):
        # =============== same in ComplexDataset ===============
        l_idx = self.data_list[idx]

        # get base feature
        dic_ligand = self.base_dataset.getitem_ligand(l_idx)

        # get scaf relative feature
        dic_ligand.update(self.base_dataset.get_scaf(l_idx))

        # reduce batch size
        dic_ligand['max_drop_size'] = None
        dic_ligand['max_drop_retain'] = None
        dic_ligand['mode'] = self.base_dataset.mode

        # =============== additional ===============
        dic_ligand['task_label'] = torch.from_numpy(self.dic_y_list[l_idx]).float()

        return dic_ligand

    def __len__(self):
        return len(self.data_list)


def collate_tasked_data(batch_list):
    dic_data = collate_ligand(batch_list)
    dic_data['task_label'] = torch.stack([g['task_label'] for g in batch_list], dim=0)
    return dic_data





if __name__ == '__main__':
    print('testing...')
    pass


