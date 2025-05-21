import numpy as np
from collections import defaultdict

import pandas as pd
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
from einops import reduce, rearrange

from ScafVAE.model.task_config import *
from ScafVAE.model.main_layers import *
from ScafVAE.model.config import *
from ScafVAE.utils.training_utils import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.common import *
from ScafVAE.utils.graph2mol import *
from ScafVAE.utils.hub import *
from ScafVAE.preprocess.som_preprocess import *
from ScafVAE.preprocess.S1_pred_ppl import Model as PPLInferModel


def prepare_frag(som_prepared_path, pred_ppl_path, idx):
    dic_data = pickle.load(open(f'{som_prepared_path}/{idx}.pkl', 'rb'))
    bond_ppl_ce = np.load(f'{pred_ppl_path}/{idx}.npz')['bond_ppl_ce']

    N = dic_data['ligand_node_features'].shape[0]
    bond_ppl_ce = torch.from_numpy(bond_ppl_ce[:N, :N])
    bond_in_ring = torch.from_numpy(dic_data['ligand_edge_features'][:N, :N, -1])
    x_label = F.pad(torch.from_numpy(dic_data['ligand_node_features']), (1, 0), 'constant', 0)[:N, :len(atom_types)].argmax(dim=-1)
    edge_label = torch.from_numpy(dic_data['ligand_edge_features'][:N, :N]).argmax(dim=-1)
    mask = torch.ones(N)

    dic_output = split_mol(
        bond_ppl_ce.unsqueeze(dim=0),
        edge_label.unsqueeze(dim=0),
        mask.unsqueeze(dim=0),
        bond_in_ring=bond_in_ring.unsqueeze(dim=0),
        bond_ppl_cutoff=PPLPredictor_config.bond_ppl_cutoff,  # WARNING: IMPORTANT FACTOR !!!
        frag_ppl_cutoff=None,
        ppl_force_max_atom=global_config.ppl_force_max_atom,  # WARNING: IMPORTANT FACTOR !!!
        max_split_mol_iter=20,
        min_components=None,
        only_single=True,
        no_ring_branch=True,
    )

    component_idx = dic_output['component_idx'][0]
    value, count = component_idx.unique(return_counts=True)
    assert count.max() <= global_config.ppl_force_max_atom_with_ring

    x = F.one_hot(x_label)
    edge = F.one_hot(edge_label)
    mol_state = get_state(x, edge, remove_dummy=True)
    is_valid, mol = quick_check(mol_state, return_with_obj=True)
    frag_mol_combined, bridge = break_bond(mol, component_idx, add_dummy=False)

    frag_mol_index_tup = Chem.GetMolFrags(frag_mol_combined, asMols=False)
    # frag_mols = Chem.GetMolFrags(frag_mol_combined, asMols=True)

    assert len(frag_mol_index_tup) <= global_config.max_frag_size

    N_cumsum = 0
    reordered_edge = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), len(bond_types)))
    reordered_edge[..., 0] = 1
    dic_mol = defaultdict(list)
    for i in range(len(frag_mol_index_tup)):
        frag_idx = np.array(frag_mol_index_tup[i])

        edge = edge_label[torch.from_numpy(frag_idx), :][:, torch.from_numpy(frag_idx)]
        all_bonds = [(bond[0].item(),
                      bond[1].item(),
                      bond_types[edge[bond[0], bond[1]].item()])
                     for bond in torch.nonzero(edge.triu(diagonal=1))]
        scaf_mol_with_original_idx = quick_check((['*'] * len(frag_idx), None, all_bonds), return_raw=True)

        scaf_smi = Chem.MolToSmiles(scaf_mol_with_original_idx, isomericSmiles=False, canonical=True)

        scaf_edge = scaf_smi_to_feat(scaf_smi)
        reordered_edge[N_cumsum:N_cumsum+scaf_edge.shape[0], N_cumsum:N_cumsum+scaf_edge.shape[0]] = scaf_edge

        match = np.array(scaf_mol_with_original_idx.GetSubstructMatches(Chem.MolFromSmiles(scaf_smi), uniquify=False, useChirality=True))[0]

        dic_mol['match'].append(match + N_cumsum)
        dic_mol['frag_idx'].append(frag_idx)

        dic_mol['reordered_component_idx'].append(np.zeros(scaf_edge.shape[0]) + i)
        dic_mol['scaf_smi'].append(scaf_smi)

        N_cumsum += scaf_edge.shape[0]
    match = np.concatenate(dic_mol['match'])
    frag_idx = np.concatenate(dic_mol['frag_idx'])
    reordered_component_idx = np.concatenate(dic_mol['reordered_component_idx'])
    frag_smi_combined = ','.join(dic_mol['scaf_smi'])

    frag_codebook = pd.read_csv(ScafVAE_config.frag_codebook_path)['scaf_smi'].tolist()
    for scaf_smi in dic_mol['scaf_smi']:
        assert scaf_smi in frag_codebook

    edge = edge_label[torch.from_numpy(frag_idx[match]), :][:, torch.from_numpy(frag_idx[match])]
    all_bonds = [(bond[0].item(),
                  bond[1].item(),
                  bond_types[edge[bond[0], bond[1]].item()])
                 for bond in torch.nonzero(edge.triu(diagonal=1))]
    reordered_mol = quick_check((['*'] * len(frag_idx), None, all_bonds), return_raw=True)

    for bond in reordered_mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if reordered_edge[a, b, 0] == 1:
            reordered_edge[a, b] = 0
            reordered_edge[b, a] = 0
            reordered_edge[a, b, 1] = 1
            reordered_edge[b, a, 1] = 1

    # edge = torch.from_numpy(reordered_edge).argmax(-1)
    # all_bonds = [(bond[0].item(),
    #               bond[1].item(),
    #               bond_types[edge[bond[0], bond[1]].item()])
    #              for bond in torch.nonzero(edge.triu(diagonal=1))]
    # scaf_mol = quick_check((['*'] * mol.GetNumAtoms(), None, all_bonds), return_raw=True)
    #
    # display(Draw.MolsToGridImage(
    #     [mol_add_idx(scaf_mol), mol_add_idx(mol)],
    #     molsPerRow=2, subImgSize=(600, 600),
    # ))
    # debug()

    dic_output = dict(
        frag_smi_combined=frag_smi_combined,
        reordered_component_idx=reordered_component_idx,
        reordered_edge=reordered_edge,
        component_idx=to_np(component_idx),
    )

    return dic_output


def process_smi(smi_list, args, device='cpu', cache='./cache', batch_size=32):
    # ================ basic feat ================
    args.max_len_ligand = global_config.max_l_len
    delmkdir(f'{cache}/basic_feat')
    succ_count = 0
    for i in trange(len(smi_list), desc='Preparing basic features'):
        tupin = (process_som, (smi_list[i], args))
        succ, data = try_do(tupin, no_try=False, break_inputs=True)

        if succ:
            pickle.dump(data, open(f'{cache}/basic_feat/{i}.pkl', 'wb'))
            succ_count += 1

    print(f'Prepared basic features: {succ_count}/{len(smi_list)} {succ_count/len(smi_list)*100:.2f}%')

    # ================ ppl ================
    model = PPLInferModel()
    model.train(False)
    model = model.to(device)

    args.data_path = f'{cache}/basic_feat'
    data_list = [i.split('.')[0] for i in os.listdir(args.data_path)]
    args.n_batch = len(data_list)

    dataset_obj = dic_global_config['dataset_obj'][args.config]
    collate_fn = dic_global_config['collate_fn_obj'][args.config]
    test_dataset = dataset_obj('eval', data_list, args)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        num_workers=2, shuffle=False,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else 0,
        collate_fn=collate_fn,
    )
    test_loader.dataset.training = False

    delmkdir(f'{cache}/ppl_pred')
    for dic_data in tqdm(test_loader, desc='Predicting bond ppl'):
        dic_data = dic2device(dic_data, device=device)
        with torch.no_grad():
            dic_output = model(dic_data)

        for i in range(len(dic_output['idx'])):
            idx = dic_output['idx'][i]
            np.savez_compressed(f'{cache}/ppl_pred/{idx}.npz', bond_ppl_ce=to_np(dic_output['bond_ppl_ce'][i]))

    # ================ get frag ================
    succ_count = 0
    delmkdir(f'{cache}/component_idx')
    for i in trange(len(smi_list), desc='Preparing fragments'):
        tupin = (prepare_frag, (f'{cache}/basic_feat', f'{cache}/ppl_pred', i))
        succ, data = try_do(tupin, no_try=False, break_inputs=True)

        if succ:
            np.savez_compressed(f'{cache}/component_idx/{i}.npz', **data)
            succ_count += 1

    print(f'Prepared mols: {succ_count}/{len(smi_list)} {succ_count/len(smi_list)*100:.2f}%')


    # ================ csv ================
    # output successfully processed data
    dic_data = defaultdict(list)
    for i in range(len(smi_list)):
        if os.path.exists(f'{cache}/component_idx/{i}.npz'):
            smi = pickle.load(open(f'{cache}/basic_feat/{i}.pkl', 'rb'))['smi']

            dic_data['idx'].append(i)
            dic_data['smi'].append(smi)

    df_data = pd.DataFrame.from_dict(dic_data)
    df_data.to_csv(f'{cache}/df_data.csv', index=False)

    del model
    torch.cuda.empty_cache()