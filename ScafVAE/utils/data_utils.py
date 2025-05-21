import networkx as nx

from einops import rearrange, repeat, reduce
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from openfold.utils.precision_utils import is_fp16_enabled

from ScafVAE.utils.common import *
from ScafVAE.utils.graph2mol import *

MAX_VALID_fp16 = 5e+4


def split_dataset(data_path, data_split_rate, test_path=None):
    id_list = [i.split('.')[0] for i in os.listdir(data_path)]

    if test_path is None:
        random.shuffle(id_list)

        l = len(id_list)
        cut_1 = int(data_split_rate[0] * l)
        cut_2 = cut_1 + int(data_split_rate[1] * l)
        train_list = id_list[:cut_1]
        val_list = id_list[cut_1:cut_2]
        test_list = id_list[cut_2:]
    else:
        test_list = [i for i in load_idx_list(test_path) if i in id_list]
        rest_list = [i for i in id_list if i not in test_list]
        random.shuffle(rest_list)

        l = len(rest_list)
        cut_2 = int(data_split_rate[1] * l)
        val_list = rest_list[:cut_2]
        train_list = rest_list[cut_2:]

    return train_list, val_list, test_list


def save_splited_dataset(train_list, val_list, test_list, save_path):
    save_idx_list(train_list, f'{save_path}/train_list.txt')
    save_idx_list(val_list, f'{save_path}/val_list.txt')
    save_idx_list(test_list, f'{save_path}/test_list.txt')


def get_max_tensor(x):
    return MAX_VALID_fp16 if is_fp16_enabled() else torch.finfo(x.dtype).max


def pad_zeros(batch_list, keys, max_len, collect_dim=-3, data_type='1d', cat=False, value=0, output_dtype=None):
    # 1d: set of [..., pad_dim, ...], 2d: set of [..., pad_dim, pad_dim + 1, ...]
    # To:
    # 1d: [..., collect_dim, pad_dim, ...], 2d: [..., collect_dim, pad_dim, pad_dim + 1, ...]
    assert collect_dim < 0
    pad_dim = collect_dim + 1

    collect = torch.concat if cat else torch.stack

    dic_data = {}
    for k in keys:
        if k not in batch_list[0].keys():
            continue

        if data_type == '1d':
            collection = collect([F.pad(g[k],
                   tuple([0] * (np.abs(pad_dim) - 1) * 2 + [0, max_len - g[k].shape[pad_dim]]),
                   'constant', value)
                   for g in batch_list], dim=collect_dim)
        if data_type == '2d':
            collection = collect([F.pad(g[k],
                   tuple([0] * (np.abs(pad_dim) - 2) * 2 + [0, max_len - g[k].shape[pad_dim]]*2),
                   'constant', value)
                   for g in batch_list], dim=collect_dim)
        else:
            assert data_type in ['1d', '2d']

        if output_dtype is not None:
            collection = collection.to(output_dtype)
        dic_data[k] = collection

    return dic_data


def simple_pad(data_list):
    # data_list [data_1, data_2, ...]
    # data_i: [N, ...]
    max_len = 0
    for data in data_list:
        max_len = max(max_len, data.shape[0])
    batched_data = torch.stack([
        F.pad(data, tuple([0] * (len(data.shape) - 1) * 2 + [0, max_len - data.shape[0]]), 'constant', 0)
        for data in data_list
    ], dim=0)
    return batched_data


def collate_dummy(batch_list):
    return batch_list[0]


def get_cosine_similarity(x1, x2, chunk=None):
    if chunk is None:
        # sim = (x1 * x2).sum(dim=-1) / (x1.norm(p=2, dim=-1) * x2.norm(p=2, dim=-1)).clamp(min=1e-5)
        sim = F.cosine_similarity(x1, x2, dim=-1)
    else:
        # chunk at x1 dim=0, x2 dim=1
        # x1,x2 = [B, N, D]
        assert x1.shape[0] != 0
        assert x2.shape[1] != 0

        sim = []
        for sub_x1 in x1.split(chunk, dim=0):
            sub_sim = []
            for sub_x2 in x2.split(chunk, dim=1):
                sub_sim.append(F.cosine_similarity(sub_x1, sub_x2, dim=-1))
            sub_sim = torch.concat(sub_sim, dim=-1)
            sim.append(sub_sim)
        sim = torch.concat(sim, dim=0)

    return sim


def shuffle_graph(x, edge, mask):
    device = x.device
    B, N = x.shape
    shuffle_idx = torch.rand(B, N, device=device)
    shuffle_idx.masked_fill_(~mask.bool(), -get_max_tensor(shuffle_idx))
    shuffle_idx = torch.argsort(shuffle_idx, dim=-1, descending=True)

    x = torch.gather(x, index=shuffle_idx, dim=-1)
    edge = torch.gather(
        edge,
        index=repeat(
            shuffle_idx, 'b i -> b i j',
            j=edge.shape[-1],
        ), dim=-2,
    )
    edge = torch.gather(
        edge,
        index=repeat(
            shuffle_idx, 'b j -> b i j',
            i=edge.shape[-2],
        ), dim=-1,
    )

    return x, edge


@torch.no_grad()
def split_mol(
        bond_ppl_ce, edge_label, mask,
        bond_in_ring=None,
        bond_ppl_cutoff=None,
        frag_ppl_cutoff=None,
        ppl_force_max_atom=None,
        max_split_mol_iter=20,
        min_components=None,
        only_single=True,
        no_ring_branch=False,
):
    '''
    split mol graph with perplexity (ppl)
    :param bond_ppl_ce: [batch, N, N], ce of each bond
    :param edge_label: [batch, N, N], edge label, see bond_types in utils.graph2mol
    :param mask: [batch, N], mask for non-dummy atoms
    :param bond_in_ring: [batch, N, N], bond in a ring
    :param bond_ppl_cutoff: cut bond if its ppl is lower than bond_ppl_cutoff
    :param frag_ppl_cutoff: split frag if its ppl is lower than frag_ppl_cutoff
    :param ppl_force_max_atom: split frag if the frag is larger than ppl_force_max_atom
    :param max_split_mol_iter: max loop for bond cut
    :param min_components: split the largest frag if the number of frags is smaller than min_components
    :param only_single: only cut single bond
    :param no_ring_branch: break bonds connected to rings
    :return:
    '''

    dic_output = defaultdict(list)
    len_list = mask.sum(dim=-1)

    for i in range(mask.shape[0]):
        N = len_list[i].int()
        adj = (edge_label[i, :N, :N] != 0).triu(diagonal=1)
        adj_origin = torch.empty_like(adj).copy_(adj)
        adj_ppl_ce = bond_ppl_ce[i, :N, :N] * adj
        adj_ppl = adj_ppl_ce.clamp(max=10.0).exp()

        # control
        adj_allow_mask = torch.empty_like(adj, dtype=torch.float).copy_(adj)
        if bond_in_ring is not None:
            ring_mask = bond_in_ring[i, :N, :N]
            adj_allow_mask = adj_allow_mask - ring_mask
        if only_single:
            single_mask = edge_label[i, :N, :N] == 1
            adj_allow_mask = adj_allow_mask * single_mask
        adj_allow_mask = adj_allow_mask.clamp(min=0)

        # initialization
        n_components = 1
        component_labels = torch.zeros(N, device=adj.device, dtype=torch.int64)

        # cut mol with bond ppl
        if bond_ppl_cutoff is not None:
            adj = torch.where((adj_ppl * adj_allow_mask > bond_ppl_cutoff).bool(), torch.zeros_like(adj), adj)
            n_components, component_labels = get_connected_components(adj)

        # cut mol with frag ppl
        if frag_ppl_cutoff is not None:
            for _ in range(max_split_mol_iter):
                value, count = component_labels.unique(return_counts=True)
                stop_flag = True
                for v, c in zip(value, count):
                    if c > 1:
                        cur_frag_mask = component_labels == v
                        frag_ppl = get_single_frag_ppl(adj_ppl_ce, adj, cur_frag_mask)
                        if frag_ppl > frag_ppl_cutoff:
                            n_components, component_labels, adj = split_frag(
                                adj_ppl, adj, cur_frag_mask, adj_allow_mask=adj_allow_mask)
                            stop_flag = False
                            break

                if stop_flag:
                    break

        # break bonds connected to rings
        if no_ring_branch and bond_in_ring is not None:
            atom_in_ring = ring_mask.sum(dim=-1).clamp(max=1)
            atom_not_in_ring = 1 - atom_in_ring
            atom_not_in_ring_pair_mask = atom_not_in_ring.unsqueeze(dim=-1) * atom_not_in_ring.unsqueeze(dim=-2)
            adj = (adj.long() - (1 - atom_not_in_ring_pair_mask) * adj_allow_mask).clamp(min=0)

            n_components, component_labels = get_connected_components(adj)

        # control frag size
        if ppl_force_max_atom is not None:
            for _ in range(max_split_mol_iter):
                value, count = component_labels.unique(return_counts=True)
                stop_flag = True
                for v, c in zip(value, count):
                    if c > ppl_force_max_atom:
                        cur_frag_mask = component_labels == v
                        n_components, component_labels, adj = split_frag(
                            adj_ppl, adj, cur_frag_mask, adj_allow_mask=adj_allow_mask)
                        stop_flag = False
                        break
                if stop_flag:
                    break

        # control min components
        if min_components is not None:
            cur_mol_min_components = min(N, min_components)

            for _ in range(max_split_mol_iter):
                if n_components >= cur_mol_min_components:
                    break

                value, count = component_labels.unique(return_counts=True)
                cur_frag_mask = component_labels == value[count.argmax()]
                n_components, component_labels, adj = split_frag(
                    adj_ppl, adj, cur_frag_mask, adj_allow_mask=adj_allow_mask)


        # get frag ppl for record
        value, count = component_labels.unique(return_counts=True)
        for v, c in zip(value, count):
            if c > 1:
                # use original adj (adj_origin), because adj was modified above
                frag_ppl_for_record = get_single_frag_ppl(adj_ppl_ce, adj_origin, component_labels == v)
                if (component_labels == v).long().sum() > 1:  # i.e. have bonds in this frag
                    dic_output['frag_ppl'].append(frag_ppl_for_record)

        # pad component_labels
        component_labels = F.pad(component_labels, (0, mask.shape[-1] - N), 'constant', n_components)

        # record, bond ppl (only allowed positions)
        allow_bond_loc = torch.nonzero(adj_origin * adj_allow_mask)
        bond_ppl = adj_ppl[allow_bond_loc[:, 0], allow_bond_loc[:, 1]]
        dic_output['bond_ppl'].append(bond_ppl[bond_ppl >= 0])  # exclude bond in ring

        dic_output['n_components'].append(n_components)
        dic_output['component_idx'].append(component_labels)
        dic_output['broken_adj'].append(adj)

    dic_output['n_components'] = torch.tensor(dic_output['n_components'], device=mask.device)
    dic_output['component_idx'] = torch.stack(dic_output['component_idx'], dim=0)
    dic_output['bond_ppl'] = torch.concat(dic_output['bond_ppl'], dim=0)
    dic_output['frag_ppl'] = torch.tensor(dic_output['frag_ppl'], device=mask.device)

    return dic_output


def split_frag(adj_ppl, adj, frag_mask, adj_allow_mask=None):
    '''
    split fragment with the highest ppl bond
    :param adj_ppl: [N, N], ppl for bonds
    :param adj: [N, N], adj matrix
    :param frag_mask: [N], frag mask
    :param adj_allow_mask: [N, N], mask for allowed positions
    :return:
    '''
    frag_mask_pair = frag_mask.unsqueeze(dim=-1) * frag_mask
    frag_adj = adj * frag_mask_pair
    max_ppl_bond = adj_ppl == (adj_ppl * frag_adj * adj_allow_mask).max()
    adj = torch.where(max_ppl_bond.bool(), torch.zeros_like(adj), adj)
    n_components, component_labels = get_connected_components(adj)
    return n_components, component_labels, adj


def get_single_frag_ppl(adj_ppl_ce, adj, frag_mask):
    '''
    get fragment perplexity with a mask
    :param adj_ppl: [N, N], ce for each bond
    :param adj: [N, N], adj matrix
    :param frag_mask: [N], mask for frag positions
    :return: fragment perplexity
    '''
    pair_mask = (frag_mask.unsqueeze(dim=-1) * frag_mask * adj).triu(diagonal=1)
    bond_count = pair_mask.sum()
    frag_ppl = ((adj_ppl_ce * pair_mask).sum() / (bond_count + EPS)).clamp(max=10.0).exp()
    return frag_ppl


def get_connected_components(adj):
    '''
    get independent fragments
    :param adj: [N, N], adj matrix
    :return:
    '''
    adj_numpy = to_np(adj)
    csgraph = csr_matrix(adj_numpy)
    n_components, labels = connected_components(csgraph=csgraph, directed=False, return_labels=True)
    labels = torch.tensor(labels, device=adj.device, dtype=torch.int64)
    return n_components, labels


def VQ_to_nx(node, edge, blank_node=0, blank_edge=0):
    node_idx = torch.nonzero((node != blank_node).float()).reshape(-1)

    edge = 1 - (edge == blank_edge).float()
    edge_idx = torch.nonzero(edge.triu(diagonal=1))

    G = nx.Graph()
    for idx in node_idx:
        G.add_node(int(idx))
    for bond_idx in edge_idx:
        G.add_edge(int(bond_idx[0]), int(bond_idx[1]))

    return G


def nx_bfs_sort(G):
    source = np.random.randint(G.number_of_nodes())
    dic_bfs = dict(nx.bfs_successors(G, source))
    idx_list = [source]
    start = [source]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dic_bfs.get(current)
            if neighbor is not None:
                next = next + neighbor
        idx_list = idx_list + next
        start = next
    return idx_list


def bfs_sort(node, edge, blank_node=0, blank_edge=0):
    G = VQ_to_nx(node, edge, blank_node=blank_node, blank_edge=blank_edge)
    idx = nx_bfs_sort(G)
    idx = torch.tensor(idx, dtype=node.dtype, device=node.device)
    idx = torch.concat([idx, torch.arange(len(idx), len(node), device=idx.device)])
    return idx


@torch.no_grad()
def get_component_inner_idx(component_idx):
    unique_frag_idx = component_idx.unique()
    unique_frag_base = torch.zeros_like(component_idx) + rearrange(unique_frag_idx, 'n -> n () ()')
    frag_hit = component_idx == unique_frag_base

    # component_idx [0, 0, 0, 1, 1] -> inner_bias_idx [0, 1, 2, 0, 1]
    inner_bias_idx = ((frag_hit.cumsum(dim=-1) - 1) * frag_hit).sum(dim=0)

    # component_idx [0, 0, 0, 1, 1] -> inner_count_idx [3, 3, 3, 2, 2]
    inner_count_idx = (frag_hit.sum(dim=-1, keepdims=True) * frag_hit).sum(dim=0)

    return inner_bias_idx, inner_count_idx


def pad_mask(x, value=0):
    return F.pad(x, (0, 1), 'constant', value)


def pad_one(x):
    return pad_mask(x, value=1)


def get_sparse_scaf(edge, component_idx, max_frag_atom):
    # edge: [N, N, D], zero for padding
    # max_frag_atom, i.e. ppl_force_max_atom
    N_scaf = component_idx.max() + 1
    scaf_adj = torch.zeros(N_scaf, N_scaf, max_frag_atom**2 + 1)
    scaf_adj[..., -1] = 1
    for i in range(N_scaf):
        for j in range(i, N_scaf):
            if i == j:
                continue

            l_mask = component_idx == i
            r_mask = component_idx == j

            sub_edge = edge[l_mask, :][:, r_mask][..., 1]
            if sub_edge.sum() == 0:
                continue

            bridge_posi = torch.nonzero(sub_edge)  # [N_bridge, 2]
            # assert bridge_posi.shape[0] <= 1, bridge_posi.shape  # only single bridge between frags
            bridge_posi = bridge_posi[0]

            bridge_posi_flat = bridge_posi[0] * max_frag_atom + bridge_posi[1]
            scaf_adj[i, j] = 0
            scaf_adj[i, j, bridge_posi_flat] = 1

            bridge_posi_flat = bridge_posi[1] * max_frag_atom + bridge_posi[0]
            scaf_adj[j, i] = 0
            scaf_adj[j, i, bridge_posi_flat] = 1

    assert (scaf_adj.sum(dim=-1) == 1).all()
    scaf_adj = scaf_adj.argmax(dim=-1)

    return scaf_adj


@torch.no_grad()
def combine_scaf(x, edge, mask, max_frag_atom, df_scaf, scaf_feat_path=None):
    # inputs are unbatched

    x = x[mask.bool()]
    edge = edge[mask.bool(), :][:, mask.bool()]

    is_pad_mask = x == 0
    x = x[~is_pad_mask]
    edge = edge[~is_pad_mask, :][:, ~is_pad_mask]

    # fill diag scaf
    N = len(x) * max_frag_atom
    N_cumsum = 0
    reordered_edge = torch.zeros(N, N, len(bond_types), device=x.device)
    reordered_edge[..., 0] = 1
    component_idx = []
    for i in range(len(x)):
        scaf_i = int(x[i] - 1)
        if scaf_feat_path is not None:
            scaf_edge = torch.from_numpy(
                np.load(f'{scaf_feat_path}/{scaf_i}.npz')['scaf_feat']
            ).to(x.device)
        else:
            scaf_edge = torch.from_numpy(scaf_smi_to_feat(df_scaf['scaf_smi'][scaf_i])).to(x.device)

        reordered_edge[N_cumsum:N_cumsum+scaf_edge.shape[0], N_cumsum:N_cumsum+scaf_edge.shape[0]] = scaf_edge
        component_idx.append(torch.empty(scaf_edge.shape[0], device=x.device, dtype=torch.long).fill_(i))

        N_cumsum += scaf_edge.shape[0]
    component_idx = torch.concat(component_idx)
    if len(component_idx) < N:
        component_idx = F.pad(component_idx, (0, N - len(component_idx)), 'constant', component_idx.max()+1)
    mask = torch.zeros(N, device=x.device)
    mask[:N_cumsum] = 1

    # fill bridge
    N_scaf = len(x)
    for i in range(N_scaf):
        for j in range(i, N_scaf):
            if i == j:
                continue

            bridge_type = int(edge[i, j])
            if bridge_type >= (max_frag_atom ** 2):  # BLANK / mask
                continue

            sub_left_posi = bridge_type // max_frag_atom
            sub_right_posi = bridge_type % max_frag_atom

            arange = torch.arange(reordered_edge.shape[1], device=edge.device)
            left_posi = arange[component_idx == i][sub_left_posi]
            right_posi = arange[component_idx == j][sub_right_posi]

            reordered_edge[left_posi, right_posi] = 0
            reordered_edge[right_posi, left_posi] = 0
            reordered_edge[left_posi, right_posi, 1] = 1
            reordered_edge[right_posi, left_posi, 1] = 1

    # reordered_edge = reordered_edge.argmax(dim=-1)

    component_idx = component_idx[mask.bool()]
    reordered_edge = reordered_edge[mask.bool(), :][:, mask.bool()]

    return reordered_edge, component_idx


def has_cycle(edge):
    edge = to_np(torch.nonzero(edge.triu(diagonal=1)))

    parent = {vertex: vertex for vertex in range(edge.max() + 1)}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for e in edge:
        if find(e[0]) == find(e[1]):
            return True
        union(e[0], e[1])
    return False


def reconst_mol(
    edge_aa, x_aa=None,
):
    # x_aa: [N]
    # edge_aa: [N, N]
    if x_aa is None:
        ele_list = ['*'] * edge_aa.shape[0]
    else:
        x_aa = x_aa[:edge_aa.shape[1]]
        ele_list = [atom_types[i.item()] for i in x_aa]
    all_bonds = [(bond[0].item(),
                  bond[1].item(),
                  bond_types[edge_aa[bond[0], bond[1]].item()])
                 for bond in torch.nonzero(edge_aa.triu(diagonal=1))]
    mol = quick_check((ele_list, None, all_bonds), return_raw=True)

    return mol


def mol2smi(mol, rescue=False, return_mol=False):
    try:
        # with suppress_stdout_stderr():
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol, canonical=True)
    except:
        smi = None

    if rescue and smi is None:
        try:
            # with suppress_stdout_stderr():
            mol = rule_base_fix(mol)
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(smi)
        except:
            smi = None

    if return_mol:
        return smi, mol
    else:
        return smi


def reconst_smi(edge_aa, x_aa, rescue=False):
    mol = reconst_mol(
        edge_aa,
        x_aa=x_aa,
    )
    smi = mol2smi(mol, rescue=rescue, return_mol=False)
    return smi



if __name__=='__main__':
    edge = torch.LongTensor(
        [
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    print(has_cycle(edge))










