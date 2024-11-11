import copy
import os

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from molecular_rectifier import Rectifier
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from ScafVAE.preprocess.bind_preprocess import get_ligand_bondfeature


atom_types = ['dummy atoms', 'UNK', 'C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'B', 'I', 'P']
bond_types = ['BLANK', Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

dic_allowed_valence = {
    'C': ['AROMATIC_AROMATIC', 'AROMATIC_AROMATIC_AROMATIC', 'AROMATIC_AROMATIC_DOUBLE', 'AROMATIC_AROMATIC_SINGLE',
          'AROMATIC_TRIPLE', 'DOUBLE', 'DOUBLE_DOUBLE', 'DOUBLE_SINGLE', 'DOUBLE_SINGLE_SINGLE', 'SINGLE',
          'SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE_SINGLE', 'SINGLE_TRIPLE', 'TRIPLE'],
    'N': ['AROMATIC_AROMATIC', 'AROMATIC_AROMATIC_AROMATIC', 'AROMATIC_AROMATIC_DOUBLE', 'AROMATIC_AROMATIC_SINGLE',
          'DOUBLE', 'DOUBLE_SINGLE', 'SINGLE', 'SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE', 'TRIPLE'],
    'O': ['AROMATIC_AROMATIC', 'DOUBLE', 'SINGLE', 'SINGLE_SINGLE'],
    'S': ['AROMATIC_AROMATIC', 'AROMATIC_AROMATIC_SINGLE', 'DOUBLE', 'DOUBLE_DOUBLE', 'DOUBLE_DOUBLE_SINGLE',
          'DOUBLE_DOUBLE_SINGLE_SINGLE', 'DOUBLE_SINGLE', 'DOUBLE_SINGLE_SINGLE', 'DOUBLE_SINGLE_SINGLE_SINGLE',
          'SINGLE', 'SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE_SINGLE',
          'SINGLE_SINGLE_SINGLE_SINGLE_SINGLE_SINGLE', 'SINGLE_TRIPLE'],
    'Cl': ['SINGLE'],
    'F': ['SINGLE'],
    'Br': ['SINGLE'],
    'P': ['AROMATIC_AROMATIC', 'AROMATIC_AROMATIC_SINGLE', 'DOUBLE', 'DOUBLE_DOUBLE_SINGLE', 'DOUBLE_SINGLE',
          'DOUBLE_SINGLE_SINGLE', 'DOUBLE_SINGLE_SINGLE_SINGLE', 'SINGLE', 'SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE',
          'SINGLE_SINGLE_SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE_SINGLE_SINGLE',
          'SINGLE_SINGLE_SINGLE_SINGLE_SINGLE_SINGLE'],
    'I': ['DOUBLE_DOUBLE_SINGLE', 'DOUBLE_SINGLE', 'DOUBLE_SINGLE_SINGLE_SINGLE', 'SINGLE', 'SINGLE_SINGLE',
          'SINGLE_SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE_SINGLE_SINGLE'],
    'B': ['AROMATIC_AROMATIC_SINGLE', 'DOUBLE_SINGLE', 'SINGLE', 'SINGLE_SINGLE', 'SINGLE_SINGLE_SINGLE'],
}


# dic_valence2atom: {'0_0_0_0_2': ['C', 'N', 'O', 'S', 'P'], ...}
dic_valence2atom = defaultdict(list)
dic_strbond2bond = {'AROMATIC': 4, 'DOUBLE': 2, 'SINGLE': 1, 'TRIPLE': 3}
for k, v in dic_allowed_valence.items():
    for t in v:
        counte_t = [0, 0, 0, 0, 0]
        for i in t.split('_'):
            counte_t[dic_strbond2bond[i]] += 1
        counte_t = '_'.join(map(str, counte_t))
        dic_valence2atom[counte_t].append(k)

# dic_valence2allowmask = {'0_0_0_0_2': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1], ...}
dic_valence2allowmask = {k: [1 if a in v else 0 for a in atom_types] for k, v in dic_valence2atom.items()}


def is_rdkitvalid(mol, check_frag=False):
    try:
        Chem.SanitizeMol(mol)
        if check_frag:
            n_frag = len(Chem.GetMolFrags(mol, asMols=False))
            if n_frag == 1:
                return True, mol
            else:
                return False, mol
        else:
            return True, mol
    except Exception as err:
        return False, err


def show_mols(mol_list, fig_save_path=None, add_idx=True):
    if add_idx:
        mol_list = [mol_add_idx(mol) for mol in mol_list]
    img = Draw.MolsToGridImage(mol_list, molsPerRow=4, subImgSize=(400, 400), legends=['' for x in mol_list])
    # plt.imshow(img)
    # if fig_save_path is not None:
    #     plt.savefig(fig_save_path, bbox_inches='tight', dpi=600)
    # plt.show()
    return img


def remove_dummy_node(x, edge, dummy_macro=False):
    # no batch
    assert len(x.shape) in [1]
    assert len(edge.shape) in [2]

    x = (x != 0) * (x != 1)
    edge = edge.sum(dim=-1) != 0

    if dummy_macro:
        no_dummy = x.float() + edge.float()
    else:
        no_dummy = x.float() * edge.float()

    no_dummy = no_dummy.bool()
    return no_dummy


def get_state(x, edge, remove_dummy=True, with_chiral=False):
    # no batch
    assert len(x.shape) in [2]
    assert len(edge.shape) in [3]
    x_ele = x[:, :len(atom_types)].argmax(dim=-1)
    if with_chiral:
        x_chiral = x[:, len(atom_types):].argmax(dim=-1)
    edge = edge.argmax(dim=-1)

    if remove_dummy:
        no_dummy = remove_dummy_node(x_ele, edge, dummy_macro=False)
        if no_dummy.any() == False:
            return (None, None, None)
        x_ele = x_ele[no_dummy]
        if with_chiral:
            x_chiral = x_chiral[no_dummy]
        edge = edge[no_dummy, :][:, no_dummy]

    ele_str_list = [atom_types[i.item()] for i in x_ele]
    if with_chiral:
        chiral_list = [Chem.ChiralType.values[i.item()] for i in x_chiral]
    else:
        chiral_list = None
    all_bonds = [(bond[0].item(),
                  bond[1].item(),
                  bond_types[edge[bond[0], bond[1]].item()])
                 for bond in torch.nonzero(edge.triu(diagonal=1))]

    return (ele_str_list, chiral_list, all_bonds)


def quick_check(
        mol_state,
        check_frag=False,
        omit_dummy_atoms=True,
        return_with_obj=True,
        with_chiral=False,
        return_raw=False,
):
    ele_str_list, chiral_list, all_bonds = mol_state

    if ele_str_list is None:
        is_valid = False
        return_obj = None

    else:
        N = len(ele_str_list)
        mol = Chem.RWMol()

        for i in range(N):
            ele = '*' if ele_str_list[i] in ['dummy atoms', 'UNK'] else ele_str_list[i]
            a = Chem.Atom(ele)
            if with_chiral:
                a.SetChiralTag(chiral_list[i])
            mol.AddAtom(a)

        for bond_1, bond_2, e in all_bonds:
            mol.AddBond(bond_1, bond_2, e)

        if return_raw:
            return mol

        if omit_dummy_atoms:
            dummy_posi = []
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetSymbol() == '*':
                    dummy_posi.append(i)

        is_valid, return_obj = is_rdkitvalid(mol, check_frag=check_frag)

        if return_with_obj and is_valid and omit_dummy_atoms:
            return_obj = Chem.DeleteSubstructs(return_obj, Chem.MolFromSmarts('[#0]'))
        if return_with_obj and is_valid:
            return_obj = update_chirality(return_obj)

    if return_with_obj:
        return (is_valid, return_obj)
    else:
        return is_valid


def break_bond(mol, component_idx, add_dummy=True):
    '''
    mol fragmentation
    :param mol: mol object
    :param component_idx: [N], given by split_mol()
    :return:
    '''
    component_mask = 0
    for v in component_idx.unique():
        sub_mask = component_idx == v
        sub_mask = sub_mask.unsqueeze(dim=-1) * sub_mask
        component_mask = component_mask + sub_mask

    bonds_need_remove = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if component_mask[a, b] == 0:
            bonds_need_remove.append(bond.GetIdx())

    emol = copy.deepcopy(mol)
    emol = Chem.FragmentOnBonds(emol, bonds_need_remove, addDummies=add_dummy)
    return emol, bonds_need_remove


def mol_add_idx(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    return mol


def draw_highlight_mol(mol, output_path, atom_list=None, bond_list=None):
    # bond_list: [N, 2]
    if bond_list is not None:
        bond_list = [mol.GetBondBetweenAtoms(aid1, aid2).GetIdx() for aid1, aid2 in bond_list]

    d = Draw.rdMolDraw2D.MolDraw2DCairo(600, 600)  # MolDraw2DSVG / MolDraw2DCairo
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_list, highlightBonds=bond_list)
    d.FinishDrawing()
    d.WriteDrawingText(output_path)

def update_chirality(mol):
    try:
        mol.UpdatePropertyCache()
    except:
        pass
    return mol


def remove_dummy(mol):
    att = Chem.MolFromSmiles('*')
    H = Chem.MolFromSmiles('[H]')
    mol = Chem.ReplaceSubstructs(mol, att, H, replaceAll=True)[0]
    mol = Chem.RemoveHs(mol)
    return mol


def replace_all_to_dummy(mol):
    mw = Chem.RWMol(mol)
    for i in range(mol.GetNumAtoms()):
        mw.ReplaceAtom(i, Chem.Atom('*'))
    mol = mw.GetMol()
    return mol


def get_scaf_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = remove_dummy(mol)
    mol = replace_all_to_dummy(mol)
    scaf_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return scaf_smi


def count_att(mol):
    att = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        ele = a.GetSymbol()
        if ele == '*':
            att.append(idx)
    return len(att)


def get_scaf_patt(mol, extra_atom='H'):
    # att -> H, rest -> *

    att = Chem.MolFromSmiles('*')
    extra_atom = Chem.MolFromSmiles(f'[{extra_atom}]')
    mol = Chem.ReplaceSubstructs(mol, att, extra_atom, replaceAll=True)[0]

    mw = Chem.RWMol(mol)
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        ele = a.GetSymbol()
        if ele == 'H':
            continue
        else:
            mw.ReplaceAtom(idx, Chem.Atom('*'))
    mol = mw.GetMol()

    return mol


def scaf_patt_smi_to_feat(scaf_patt_smi):
    mol = Chem.MolFromSmiles(scaf_patt_smi)

    ele = [a.GetSymbol() for a in mol.GetAtoms()]
    non_att_mask = np.array([0 if e == 'H' else 1 for e in ele])

    edge = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), len(bond_types)))
    for i in range(mol.GetNumAtoms()):
        for j in range(i, mol.GetNumAtoms()):
            feat = get_ligand_bondfeature(mol, i, j)[:-1]
            edge[i, j] = feat
            edge[j, i] = feat

    return edge, non_att_mask


def scaf_smi_to_feat(scaf_smi):
    mol = Chem.MolFromSmiles(scaf_smi)

    edge = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), len(bond_types)))
    for i in range(mol.GetNumAtoms()):
        for j in range(i, mol.GetNumAtoms()):
            feat = get_ligand_bondfeature(mol, i, j)[:-1]
            edge[i, j] = feat
            edge[j, i] = feat

    return edge


def map_valence_to_mask(valence):
    B, N = valence.shape[:2]
    valence = rearrange(valence, 'b n d -> (b n) d')

    valence_list = valence.long().tolist()
    valence_list = map(lambda x1: '_'.join(map(lambda x2: str(x2), x1)), valence_list)
    def try_get_mask(x):
        if x in dic_valence2allowmask.keys():
            return dic_valence2allowmask[x]
        else:
            return [1] + [0] * (len(atom_types) - 1)
    allow_mask = list(map(try_get_mask, valence_list))
    allow_mask = torch.tensor(allow_mask, dtype=torch.float, device=valence.device)
    allow_mask = rearrange(allow_mask, '(b n) d -> b n d', b=B)
    return allow_mask


def rule_base_fix(mol, iteration=0):
    recto = Rectifier(mol)
    recto.log.disabled = True
    recto.fix(iteration=iteration)
    fixed_mol = recto.mol
    return fixed_mol


def std_smi(smi):
    if smi is not None:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return smi


def canonicalize_smi_list(smi_list):
    canonicalized_smi_list = []
    for smi in tqdm(smi_list, desc='Canonicalizing'):
        smi = std_smi(smi)
        canonicalized_smi_list.append(smi)
    return canonicalized_smi_list














