import os
import sys
import shutil

import argparse
import yaml
from collections import Iterable
import numpy as np
import random
import pickle
from rdkit import Chem

import torch
import torch.nn.functional as F
import torch_geometric


EPS = 1e-8


def delmkdir(path, remove_old=True, create_new=True):
    isexist = os.path.exists(path)
    if isexist and remove_old:
        shutil.rmtree(path)

    isexist = os.path.exists(path)
    if not isexist and create_new:
        os.makedirs(path)


def try_do(intup, no_try=False, break_inputs=False):
    f, task = intup

    if no_try:
        if break_inputs:
            return True, f(*task)
        else:
            return True, f(task)

    try:
        if break_inputs:
            return True, f(*task)
        else:
            return True, f(task)
    except:
        return False, None


def summarize_model(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_full_number(profile='full', precision=3, sci_mode=False, linewidth=160):
    # Debug
    np.set_printoptions(suppress=True)
    torch.set_printoptions(profile=profile, precision=precision, sci_mode=sci_mode, linewidth=linewidth)


def num_list_to_str(x, cut='-'):
    def to_int_str(i):
        return str(int(i))
    return cut.join(map(to_int_str, x))


def str_to_num_list(x, cut='-'):
    return list(map(int, x.split(cut)))


def to_numpy(x):
    return x.detach().cpu().numpy()


def fix_bool(args):
    args_dict = vars(args)
    new_dic = {}
    for k, v in args_dict.items():
        if isinstance(v, str):
            if v.upper() == 'TRUE':
                v = True
            elif v.upper() == 'FALSE':
                v = False
            elif v.upper() == 'NONE':
                v = None
        new_dic[k] = v
    return argparse.Namespace(**new_dic)


def split_rate(data_split_rate):
    split_str = None
    for i in ['-', '_', ',']:
        if i in data_split_rate:
            split_str = i
    assert split_str != None
    data_split_rate = list(map(lambda x: float(x), data_split_rate.split(split_str)))
    assert np.array(data_split_rate).sum() == 1
    return data_split_rate


def print_args(args):
    print('=' * 30 + ' Current settings ' + '=' * 30)
    for k, v in args.__dict__.items():
        print(k.ljust(40, '.'), v)
    print('=' * (60 + len(' Current settings ')))


def show_extra_eval(name, value):
    print(f'Extra eval - {name}: {value}')


def load_idx_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line[:-1] if line[-1] == '\n' else line for line in lines]
    return lines


def save_idx_list(idx_list, file_path):
    with open(file_path, 'w') as f:
        f.write('\n'.join(idx_list) + '\n')


def save_val(data, rank, f_name='tmp_val'):
    with open(f'{f_name}_{rank}.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_val(f_name='tmp_val'):
    f_list = [f for f in os.listdir() if f.startswith(f_name)]

    for i, f_name in enumerate(f_list):
        with open(f_name, 'rb') as f:
            dic_tmp = pickle.load(f)
        if i == 0:
            dic = dic_tmp
        else:
            for k in dic.keys():
                dic[k] = dic[k] + dic_tmp[k]
        os.remove(f_name)
    return dic


def is_tensor(x):
    if isinstance(x, torch.Tensor) or \
       isinstance(x, torch.LongTensor) or \
       isinstance(x, torch.FloatTensor) or \
       isinstance(x, torch.BoolTensor) or \
       isinstance(x, torch.HalfTensor) or \
       isinstance(x, torch_geometric.data.data.Data):
        return True
    else:
        return False


def get_sche(sche, step=-1):
    if isinstance(sche, Iterable):
        if step >= len(sche) or step == -1:
            return sche[-1]
        else:
            return sche[step]
    else:
        return sche


def load_act(t):
    if t == 'relu':
        return F.relu
    elif t == 'leakyrelu':
        return F.leaky_relu
    elif t == 'selu':
        return F.selu
    elif t == 'tanh':
        return F.tanh
    elif t == 'elu':
        return F.elu
    else:
        raise ValueError(f'Unknown activate function {t}')


def print_dic_data(dic_data):
    for k, v in dic_data.items():
        if is_tensor(v):
            if isinstance(v, torch_geometric.data.data.Data):
                print(k, v)
            else:
                print(k, list(v.shape))
        elif isinstance(v, np.ndarray):
            print(k, list(v.shape))
        elif isinstance(v, Iterable):
            print(k, len(v))


def is_notebook():
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class suppress_stdout_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def read_rdkit_mol(mol, silence=False):
    if mol.endswith('mol2'):
        mol = Chem.MolFromMol2File(mol)
    elif mol.endswith('pdb'):
        mol = Chem.MolFromPDBFile(mol)
    elif mol.endswith('mol'):
        mol = Chem.MolFromMolFile(mol)
    elif mol.endswith('sdf'):
        SD = Chem.SDMolSupplier(mol)
        mol = [x for x in SD][0]
    else:
        mol = Chem.MolFromSmiles(mol)
    return mol


def get_module(model, args):
    if args.use_multi_gpu:
        module = model.module
    else:
        module = model
    return module


def debug(p=None):
    if p is not None:
        print(p)
    print('='*20, 'TESTED', '='*20)
    sys.exit()


def load_eval_loss(dic_loss):
    dic_eval = {}
    for k, v in dic_loss.items():
        if k.endswith('_loss'):
            k = k[:-5] + '_metric'
        dic_eval[k] = v
    return dic_eval


def np_concat(x: list):
    if isinstance(x[0], np.ndarray):
        return np.concatenate(x, axis=0)
    else:
        new_x = []
        for i in x:
            new_x += i
        return new_x


def save_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


if __name__ == '__main__':
    # for testing
    pass
