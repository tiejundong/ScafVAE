import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from biopandas.pdb import PandasPdb


def onehot_with_allowset(x, allowset, with_unk=True):
    if x not in allowset and with_unk == True:
        x = allowset[0]  # UNK
    return list(map(lambda s: x == s, allowset))


def get_mol_posi(mol, idx=-1):
    mol_conf = mol.GetConformer(id=idx)
    node_posi = np.array([mol_conf.GetAtomPosition(int(idx)) for idx in range(mol.GetNumAtoms())])
    return node_posi


def get_mol_all_posi(mol):
    node_posi_list = []
    for i, mol_conf in enumerate(mol.GetConformers()):
        node_posi_list.append(np.array([mol_conf.GetAtomPosition(int(idx)) for idx in range(mol.GetNumAtoms())]))
    return node_posi_list


def get_ligand_atomfeature(atom, idx, ring_info):
    atom_features = \
        onehot_with_allowset(atom.GetSymbol(), ['UNK', 'C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'B', 'I', 'P'], with_unk=True) + \
        onehot_with_allowset(atom.GetTotalDegree(), ['UNK', 1, 2, 3, 4, 5], with_unk=True) + \
        onehot_with_allowset(atom.GetFormalCharge(), ['UNK', -1, -2, 0, 1, 2], with_unk=True) + \
        onehot_with_allowset(atom.GetImplicitValence(), ['UNK', 0, 1, 2, 3], with_unk=True) + \
        onehot_with_allowset(atom.GetTotalNumHs(), ['UNK', 0, 1, 2, 3], with_unk=True) + \
        onehot_with_allowset(atom.GetHybridization(), \
                             ['UNK',
                              Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2], with_unk=True) + \
        [atom.GetIsAromatic()] + \
        [ring_info.IsAtomInRingOfSize(idx, 3),
         ring_info.IsAtomInRingOfSize(idx, 4),
         ring_info.IsAtomInRingOfSize(idx, 5),
         ring_info.IsAtomInRingOfSize(idx, 6),
         ring_info.IsAtomInRingOfSize(idx, 7),
         ring_info.IsAtomInRingOfSize(idx, 8)] + \
        onehot_with_allowset({v: k for k, v in Chem.rdchem.ChiralType.values.items()}[atom.GetChiralTag()],
                         list(range(len(Chem.rdchem.ChiralType.values))), with_unk=False)
    atom_features = np.array(atom_features).astype(np.float32)
    return atom_features


def get_ligand_nodefeature(mol):
    ring_info = mol.GetRingInfo()
    node_features = [get_ligand_atomfeature(atom, idx, ring_info) for idx, atom in
                     zip(range(mol.GetNumAtoms()), mol.GetAtoms())]
    node_features = np.stack(node_features)
    return node_features


def get_fullconnectedge(mol):  # include a->b, a<-b and a->a
    edge = np.array([[i, j] for i in range(mol.GetNumAtoms()) for j in range(mol.GetNumAtoms())])
    return edge


def get_havebond(mol, idx_1, idx_2):
    bond = mol.GetBondBetweenAtoms(int(idx_1), int(idx_2))
    if bond == None:
        return 0
    else:
        return 1


def get_atomdistance(mol_conf, idx_1, idx_2):
    coor_1 = mol_conf.GetAtomPosition(int(idx_1))
    coor_2 = mol_conf.GetAtomPosition(int(idx_2))
    return coor_1.Distance(coor_2)


def HasCommonNeighborAtom(mol, idx_1, idx_2):
    flag = False
    for idx in range(mol.GetNumAtoms()):
        if idx == idx_1 or idx == idx_2:
            continue
        else:
            bond_1 = mol.GetBondBetweenAtoms(int(idx), int(idx_1))
            bond_2 = mol.GetBondBetweenAtoms(int(idx), int(idx_2))
            if bond_1 != None and bond_2 != None:
                flag = True
                break
    return flag


def get_ligand_bondfeature(mol, idx_1, idx_2):
    bond = mol.GetBondBetweenAtoms(int(idx_1), int(idx_2))
    if bond == None:
        edge_feature = [1] + [0] * 5
    else:
        edge_feature = [0]
        edge_feature += onehot_with_allowset(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, \
                                                                  Chem.rdchem.BondType.DOUBLE, \
                                                                  Chem.rdchem.BondType.TRIPLE, \
                                                                  Chem.rdchem.BondType.AROMATIC], with_unk=False)
        edge_feature += [bond.IsInRing()]
    edge_feature = np.array(edge_feature).astype(np.float32)
    return edge_feature


def get_ligand_edgefeature(mol):
    edge_features = np.array([get_ligand_bondfeature(mol, i, j) for i, j in get_fullconnectedge(mol)])
    edge_features = edge_features.reshape(mol.GetNumAtoms(), mol.GetNumAtoms(), -1)
    return edge_features


def read_mol_with_pdb_smi(pdb_path, smiles):
    ligand_mol = Chem.MolFromPDBFile(pdb_path)
    ligand_template = Chem.MolFromSmiles(smiles)
    ligand_mol = AllChem.AssignBondOrdersFromTemplate(ligand_template, ligand_mol)
    assert ligand_mol != None
    return ligand_mol


def read_pdbbind_ligand(pdbbind_path, pdb_id):
    ligand_mol2_path = f'{pdbbind_path}/{pdb_id}/{pdb_id}_ligand.mol2'
    ligand_mol = Chem.MolFromMol2File(ligand_mol2_path)
    if ligand_mol == None:
        ligand_pdb_path = f'{pdbbind_path}/{pdb_id}/{pdb_id}_ligand.pdb'
        ligand_smiles_path = f'{pdbbind_path}/{pdb_id}/{pdb_id}_ligand.smi'
        ligand_smiles = open(ligand_smiles_path, 'r').readlines()[0].split('\t')[0]
        ligand_mol = read_mol_with_pdb_smi(ligand_pdb_path, ligand_smiles)
    return ligand_mol


def get_ligand_match(mol):
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True)
    return np.array(matches)


def get_ligand_unrotable_distance(ligand_mol):
    rot_patt = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'  # '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]' for rotable bond
    patt = Chem.MolFromSmarts(rot_patt)
    hit_bonds = ligand_mol.GetSubstructMatches(patt)
    em = Chem.EditableMol(ligand_mol)
    for (idx1, idx2) in hit_bonds:
        em.RemoveBond(idx1, idx2)
    p = em.GetMol()
    part_list = Chem.GetMolFrags(p, asMols=False)

    added_part_list = []
    for part in part_list:
        tmp = list(part)
        for bonds in hit_bonds:
            i, j = bonds
            if i in part:
                tmp.append(j)
            elif j in part:
                tmp.append(i)
        added_part_list.append(tmp)

    n_atoms = ligand_mol.GetNumAtoms()
    dist_map = np.zeros((n_atoms, n_atoms)) + -1
    mol_conf = ligand_mol.GetConformer()
    for part in added_part_list:
        for i in part:
            for j in part:
                dist_map[i, j] = get_atomdistance(mol_conf, i, j)
    return dist_map


def process_ligand(ligand_mol):
    ligand_node_features = get_ligand_nodefeature(ligand_mol)
    ligand_edge_features = get_ligand_edgefeature(ligand_mol)
    ligand_coor_true = get_mol_posi(ligand_mol)
    ligand_match = get_ligand_match(ligand_mol)
    ligand_dismap = get_ligand_unrotable_distance(ligand_mol)
    ligand_CR = list(Chem.CanonicalRankAtoms(ligand_mol, breakTies=False))

    dic_ligand = dict(
        ligand_node_features=ligand_node_features,
        ligand_edge_features=ligand_edge_features,
        ligand_coor_true=ligand_coor_true,
        ligand_match=ligand_match,
        ligand_dismap=ligand_dismap,
        ligand_CR=ligand_CR,
    )
    return dic_ligand


def process_som(smi, args):
    max_len_ligand = args.max_len_ligand

    # ================= ligand encoding =================
    ligand_mol = Chem.MolFromSmiles(smi)
    AllChem.EmbedMolecule(ligand_mol, maxAttempts=10, useRandomCoords=True)
    dic_ligand = process_ligand(ligand_mol)
    assert 1 < len(dic_ligand['ligand_node_features']) <= max_len_ligand, \
        f'Current model was only trained with molecules with atoms less than {max_len_ligand}'

    # ================= MMFF =================
    dic_ligand['ligand_MMFF'] = None

    # ================= smi =================
    dic_ligand['smi'] = smi

    return dic_ligand

















