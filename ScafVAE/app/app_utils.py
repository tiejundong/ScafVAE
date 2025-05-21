import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from dataclasses import dataclass
from IPython import get_ipython
from pandarallel import pandarallel
from keras_progbar import Progbar
from multiprocessing.pool import ThreadPool
import geatpy as ea
from pymoo.mcdm.pseudo_weights import PseudoWeights

from ScafVAE.utils.common import *
from ScafVAE.utils.training_utils import *
from ScafVAE.model.task_config import *
# from ScafVAE.preprocess.bind_preprocess import process_som
from ScafVAE.preprocess.S1_pred_ppl import Model as PPLModel
from ScafVAE.preprocess.prepare_input import prepare_frag as prepare_frag_with_defined_set
from ScafVAE.app.sur_utils import *
from ScafVAE.app.opt_utils import *



def restart_notebook():
    if get_ipython():
        get_ipython().kernel.do_shutdown(restart=True)


def get_demo_smi():
    smi_path = f'{root_path}/demo/demo_data/demo_smi.txt'
    data = load_idx_list(smi_path)
    return data


def prepare_demo_smi_feat(cache_path, smi_list):
    output_path = f'{cache_path}/feat'
    delmkdir(output_path)

    def prepare_demo_smi_feat_single(idx, smi):
        dic_data = process_som(smi)
        dic_data['smi'] = smi
        if len(dic_data['ligand_node_features']) <= global_config.max_l_len:
            pickle.dump(dic_data, open(f'{output_path}/{idx}.pkl', 'wb'))

    for idx in trange(len(smi_list), desc='Preparing molecular features'):
        smi = smi_list[idx]

        try_do((prepare_demo_smi_feat_single, (idx, smi)), break_inputs=True)


def prepare_demo_ppl(cache_path, device='cpu', batch_size=32):
    feat_path = f'{cache_path}/feat'
    output_path = f'{cache_path}/ppl'
    delmkdir(output_path)

    chk = load_PPLPredictor()
    args = chk['args']
    model = PPLModel()
    model.train(False)
    model = model.to(device)

    assert training_config.data.shuffle_input == False

    train_list, val_list, test_list = split_dataset(feat_path, [0.7, 0.1, 0.2])
    save_splited_dataset(train_list, val_list, test_list, cache_path)
    args.n_batch = -1
    args.is_main_process = False

    dataset = PPLDataset('test', args, data_path=feat_path, data_list=cache_path, name='custom dataset')
    dataset.data_list = dataset.sub_data_list = train_list + val_list + test_list
    dataset.n_batch = len(dataset.data_list)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size,
        num_workers=4, shuffle=False,
        collate_fn=collate_ligand,
    )
    dataloader.dataset.training = False
    assert training_config.data.shuffle_input == False

    for dic_data in tqdm(dataloader):
        dic_data = dic2device(dic_data, device=device)
        with torch.no_grad():
            dic_output = model(dic_data)

        for i in range(len(dic_output['idx'])):
            idx = dic_output['idx'][i]
            np.savez_compressed(f'{output_path}/{idx}.npz', bond_ppl_ce=to_np(dic_output['bond_ppl_ce'][i]))


def prepare_demo_scaf(cache_path, nb_workers=8):
    feat_path = f'{cache_path}/feat'
    ppl_path = f'{cache_path}/ppl'
    output_path = f'{cache_path}/scaf'
    delmkdir(output_path)

    def prepare_demo_scaf_single(idx):
        dic_data = pickle.load(open(f'{feat_path}/{idx}.pkl', 'rb'))
        bond_ppl_ce = np.load(f'{ppl_path}/{idx}.npz')['bond_ppl_ce']

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
            reordered_edge[N_cumsum:N_cumsum + scaf_edge.shape[0], N_cumsum:N_cumsum + scaf_edge.shape[0]] = scaf_edge

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

        np.savez_compressed(f'{output_path}/{idx}.npz', **dic_output)

        return dic_mol['scaf_smi']

    dic_save = defaultdict(list)
    for f_name in tqdm(os.listdir(ppl_path), desc='Preparing scaffolds'):
        idx = f_name.split('.')[0]
        r, scaf_smi = try_do((prepare_demo_scaf_single, idx), break_inputs=False)
        if r:
            dic_save['scaf_smi'] += scaf_smi

    dic_save['scaf_smi'] = list(set(dic_save['scaf_smi']))

    uniq_scaf_path = f'{cache_path}/unique_scaf_max{PPLPredictor_config.ppl_force_max_atom}_cutoff{PPLPredictor_config.bond_ppl_cutoff}_no_ring_branch.csv'

    pandarallel.initialize(progress_bar=False, nb_workers=nb_workers)
    df_frag = pd.DataFrame.from_dict(dic_save)
    df_frag = df_frag.dropna()
    df_frag['frag_mol'] = df_frag['scaf_smi'].parallel_apply(lambda x: Chem.MolFromSmiles(x))
    df_frag['frag_size'] = df_frag['frag_mol'].apply(lambda x: x.GetNumAtoms())
    df_frag[['scaf_smi', 'frag_size']].to_csv(uniq_scaf_path, index=False)

    print(f'Unique bond scaffold: {len(df_frag)}')

    def count_att(idx):
        dic_data = np.load(f'{output_path}/{idx}.npz')

        frag_smi = str(dic_data['frag_smi_combined']).split(',')
        reordered_edge = dic_data['reordered_edge']
        reordered_component_idx = dic_data['reordered_component_idx']

        dic_output = defaultdict(list)
        for i in range(len(frag_smi)):
            component_i = reordered_component_idx == i
            component_i_rest = reordered_component_idx != i

            att_count = (reordered_edge[..., 1] * component_i_rest).sum(-1)[component_i]

            dic_output[frag_smi[i]].append(att_count)

        return dic_output

    dic_save = defaultdict(list)
    for f_name in tqdm(os.listdir(output_path), desc='Counting scaffolds'):
        idx = f_name.split('.')[0]
        dic_output = count_att(idx)
        for k, v in dic_output.items():
            dic_save[k] += v

    df_frag = pd.read_csv(uniq_scaf_path)
    scaf_smi_list = df_frag['scaf_smi'].values

    all_att_num = []
    for k in scaf_smi_list:
        att_num = np.zeros(global_config.ppl_force_max_atom_with_ring)

        if k in dic_save.keys():
            v = dic_save[k]

            v = np.stack(v, 0)
            v = reduce(v, 's n -> n', 'max')
            att_num[:len(v)] = v
        else:
            att_num = att_num + 5

        all_att_num.append(att_num)
    all_att_num = np.stack(all_att_num, 0)

    scaf_att_num_path = f'{cache_path}/scaf_att_num_max{PPLPredictor_config.ppl_force_max_atom}_cutoff{PPLPredictor_config.bond_ppl_cutoff}_no_ring_branch.npz'
    np.savez_compressed(scaf_att_num_path, all_att_num=all_att_num)


def train_toy_model(
    cache_path,
    device='cpu',
    batch_size=64,
    n_epoch=200,
    lr=1e-3,
    lr_warmup_steps=20,
):
    train_list = load_idx_list(f'{cache_path}/train_list.txt')
    val_list = load_idx_list(f'{cache_path}/val_list.txt')
    test_list = load_idx_list(f'{cache_path}/test_list.txt')

    train_list, val_list, test_list = map(
        lambda x: [i for i in x if os.path.isfile(f'{cache_path}/scaf/{i}.npz')],
        (train_list, val_list, test_list)
    )
    save_splited_dataset(train_list, val_list, test_list, cache_path)

    chk = load_ModelBase()
    ScafVAE_args = chk['args']
    ScafVAE_args.n_batch = -1
    ScafVAE_args.is_main_process = False
    ScafVAE_args.rand_inp = False

    dataset = ScafDataset(
        'train', ScafVAE_args,
        data_path=f'{cache_path}/feat', data_list=cache_path, scaf_path=f'{cache_path}/scaf',
        name='Custom training',
    )
    dataset.data_list = dataset.sub_data_list = train_list
    dataset.n_batch = len(dataset.data_list)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size,
        num_workers=4, shuffle=True,
        collate_fn=collate_ligand,
    )
    dataloader.dataset.training = False

    model = ScafVAEBase().to(device)

    ScafVAE_args.lr = lr
    ScafVAE_args.lr_warmup_steps = lr_warmup_steps
    ScafVAE_args.len_train_loader = len(dataloader)
    ScafVAE_args.n_epoch = n_epoch

    param = get_default_params(model)
    param = [
        {
            'params':
                param,
            'lr': ScafVAE_args.lr,
        },
    ]
    optimizer = torch.optim.AdamW(param, lr=ScafVAE_args.lr)  # Adam
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=ScafVAE_args.lr_warmup_steps,
        num_training_steps=ScafVAE_args.len_train_loader * ScafVAE_args.n_epoch,
    )

    for epoch in range(n_epoch):
        print(f"Epoch: {epoch + 1}, lr: {optimizer.param_groups[0]['lr']:.2e}")
        progBar = Progbar(len(dataloader))
        for i, dic_data in enumerate(dataloader):
            dic_data = dic2device(dic_data, device, epoch=epoch)
            dic_data, grad_loss, dic_log = model(dic_data, epoch=epoch)

            optimizer.zero_grad()
            grad_loss[0].backward()
            optimizer.step()

            scheduler.step()

            progBar.update(i + 1, [*[(k, np.around(v, 5)) for k, v in dic_log.items()]])

    return model, ScafVAE_args


def get_demo_properties(max_num=1000, name='binding'):
    if name == 'binding':
        df_EGFR = pd.read_csv(f'{root_path}/demo/demo_data/binding/df_EGFR.csv')
        df_HER2 = pd.read_csv(f'{root_path}/demo/demo_data/binding/df_HER2.csv')
    elif name == 'docking':
        df_EGFR = pd.read_csv(f'{root_path}/demo/demo_data/docking/df_EGFR.csv')
        df_HER2 = pd.read_csv(f'{root_path}/demo/demo_data/docking/df_HER2.csv')
    else:
        assert name in ['binding', 'docking']

    df_EGFR = df_EGFR.sample(frac=1, random_state=42).iloc[:max_num]
    df_HER2 = df_HER2.sample(frac=1, random_state=42).iloc[:max_num]

    return df_EGFR, df_HER2


def prepare_prop_data(generation_path, task_list, device='cpu', batch_size=32, n_pool=32):
    # PPL
    chk = load_PPLPredictor()
    ppl_args = chk['args']
    ppl_model = PPLModel().to(device)
    ppl_model.train(False)

    assert training_config.data.shuffle_input == False

    # base model
    ScafVAE_model = ScafVAEBase().to(device)
    ScafVAE_model.train(False)
    chk = load_ModelBase()
    ScafVAE_args = chk['args']
    ScafVAE_model.load_state_dict(chk['model_state_dict'])

    # pool
    pool = ThreadPool(n_pool)

    def prepare_prop_data_single(task):
        name = task['name']
        df_data = task['data']

        task_path = f'{generation_path}/{name}'
        delmkdir(task_path)

        # =========================================================
        feat_path = f'{task_path}/feat'
        delmkdir(feat_path)
        def prepare_demo_smi_feat_single(idx, smi):
            dic_data = process_som(smi)
            dic_data['smi'] = smi
            if len(dic_data['ligand_node_features']) <= global_config.max_l_len:
                pickle.dump(dic_data, open(f'{feat_path}/{idx}.pkl', 'wb'))
        for idx, smi in tqdm(
            zip(df_data['idx'].tolist(), df_data['smi'].tolist()),
            desc=f'Preparing {name} molecular features',
            total=len(df_data),
        ):
            try_do((prepare_demo_smi_feat_single, (idx, smi)), break_inputs=True)

        # =========================================================
        ppl_path = f'{task_path}/ppl'
        delmkdir(ppl_path)

        train_list, val_list, test_list = split_dataset(feat_path, [0.7, 0.1, 0.2])
        save_splited_dataset(train_list, val_list, test_list, task_path)
        ppl_args.n_batch = -1
        ppl_args.is_main_process = False

        dataset = PPLDataset('test', ppl_args, data_path=feat_path, data_list=task_path, name=name)
        dataset.data_list = dataset.sub_data_list = train_list + val_list + test_list
        dataset.n_batch = len(dataset.data_list)
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size,
            num_workers=4, shuffle=False,
            collate_fn=collate_ligand,
        )
        dataloader.dataset.training = False

        for dic_data in tqdm(dataloader, desc=f'Calculating {name} bond perplexity'):
            dic_data = dic2device(dic_data, device=device)
            with torch.no_grad():
                dic_output = ppl_model(dic_data)
            ppl = to_np(dic_output['bond_ppl_ce'])
            def save_single(i):
                idx = dic_output['idx'][i]
                np.savez_compressed(f'{ppl_path}/{idx}.npz', bond_ppl_ce=ppl[i])
            pool.map(save_single, range(len(dic_output['idx'])))

        # =========================================================
        scaf_path = f'{task_path}/scaf'
        delmkdir(scaf_path)

        def prepare_scaf_single(idx):
            dic_scaf = prepare_frag_with_defined_set(feat_path, ppl_path, idx)
            np.savez_compressed(f'{scaf_path}/{idx}.npz', **dic_scaf)

        for idx in tqdm(
            df_data['idx'].tolist(),
            desc=f'Preparing {name} bond scaffold',
            total=len(df_data),
        ):
            try_do((prepare_scaf_single, idx), break_inputs=False)

        # =========================================================
        latent_path = f'{task_path}/latent'
        delmkdir(latent_path)

        train_list, val_list, test_list = map(
            lambda x: [i for i in x if os.path.isfile(f'{task_path}/scaf/{i}.npz')],
            (train_list, val_list, test_list)
        )
        save_splited_dataset(train_list, val_list, test_list, task_path)

        ScafVAE_args.n_batch = -1
        ScafVAE_args.is_main_process = False
        dataset = ScafDataset(
            'test', ScafVAE_args,
            data_path=f'{task_path}/feat', data_list=task_path, scaf_path=f'{task_path}/scaf',
            name=name,
        )
        dataset.data_list = dataset.sub_data_list = train_list + val_list + test_list
        dataset.n_batch = len(dataset.data_list)
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size,
            num_workers=4, shuffle=False,
            collate_fn=collate_ligand,
        )
        dataloader.dataset.training = False

        for dic_data in tqdm(dataloader, desc=f'Calculating {name} latent vectors'):
            dic_data = dic2device(dic_data, device=device)
            with torch.no_grad():
                dic_data = ScafVAE_model.frag_encoder(dic_data)
                aug_embed = ScafVAE_model.noise2repr(dic_data['noise'])
            aug_embed = to_np(aug_embed)
            def save_single(i):
                idx = dic_data['idx'][i]
                np.savez_compressed(f'{latent_path}/{idx}.npz', aug_embed=aug_embed[i])
            pool.map(save_single, range(len(dic_data['idx'])))

    for task in task_list:
        print(f"Preparing {task['name']} ...")
        assert len(task['data']) >= 32, f"Dataset {task['name']} is too small :("
        prepare_prop_data_single(task)
        print(f"{task['name']} done.")

    return ScafVAE_model


def train_surrogate_model(generation_path, task_list, n_jobs=-1, seed=42, n_cv=5, verbose=3):
    dic_output = {}

    def train_single_sur_model(task):
        name = task['name']
        task_type = task['task_type']
        df_data = task['data']
        ML_model = task['ML_model']

        task_path = f'{generation_path}/{name}'

        test_list = load_idx_list(f'{task_path}/test_list.txt')

        dic_label = {str(k): float(v) for k, v in zip(df_data['idx'], df_data['label'])}
        data = []
        label = []
        train_test_group = []
        for f_name in os.listdir(f'{task_path}/latent'):
            aug_embed = np.load(f'{task_path}/latent/{f_name}')['aug_embed']
            data.append(aug_embed)
            label.append(dic_label[f_name.split('.')[0]])
            if f_name.split('.')[0] in test_list:
                train_test_group.append(1)
            else:
                train_test_group.append(0)
        data = np.stack(data, axis=0)
        label = np.array(label)
        train_test_group = np.array(train_test_group)

        x_train = data[train_test_group==0]
        y_train = label[train_test_group==0]
        x_test = data[train_test_group==1]
        y_test = label[train_test_group==1]

        automl = AutoML(
            task_type,
            ML_model,
            verbose=verbose,
            n_jobs=n_jobs,
            seed=seed,
            n_cv=n_cv,
            balance=True,
        )
        automl.train(x_train, y_train, x_test, y_test)

        dic_ML = dict(
            automl=automl,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        dic_output[name] = dic_ML

    for task in task_list:
        print(f"Training surrogate model for {task['name']} ...")
        train_single_sur_model(task)
        print(f"{task['name']} done.")

    return dic_output


def prepare_data_and_train(generation_path, tasks, device='cpu', batch_size=32, n_jobs=16, n_pool=16):
    delmkdir(generation_path)
    ScafVAE_model = prepare_prop_data(generation_path, tasks, device=device, batch_size=batch_size, n_pool=n_pool)
    dic_surrogate_model = train_surrogate_model(generation_path, tasks, n_jobs=n_jobs)
    return ScafVAE_model, dic_surrogate_model


def select_mol(res, weights):
    F = res['ObjV']
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

    select_i = PseudoWeights(np.array(weights)).do(nF)

    return select_i


def generate_mol(n_gen, generation_path, tasks, base_model, surrogate_model,
    seed=42, bound=3.0, multi_cpu=False, n_pool=32, max_chunk_size=8, device='cpu',
    Encoding='RI',
    NIND=200,  # number of individuals
    MAXGEN=200,  # max iteration number
    verbose=False,
):
    problem = OptProblem(
        'Multi-objective molecular design with ScafVAE',
        base_model,
        [surrogate_model[task['name']]['automl'] for task in tasks],
        [1 if task['optimization_type'] == 'min' else -1 for task in tasks],
        bound=bound,
        multi_cpu=multi_cpu,
        n_pool=n_pool,
        max_chunk_size=max_chunk_size,
        device=device,
    )

    Encoding = Encoding
    NIND = NIND
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    solver = ea.moea_NSGA3_templet if len(tasks) > 3 else ea.moea_NSGA2_templet
    myAlgorithm = solver(
        problem,
        population,
    )
    myAlgorithm.MAXGEN = MAXGEN
    myAlgorithm.recOper = ea.Recsbx(XOVR=0.9, n=20, Parallel=False, Half_N=False)  # if using MOEAD, Half_N=True
    myAlgorithm.mutOper = ea.Mutpolyn(Pm=0.5, DisI=20, Parallel=False)
    myAlgorithm.logTras = 10  # frequency of logging
    myAlgorithm.verbose = verbose
    myAlgorithm.drawing = 0

    pseudo_weights = [task['pseudo_weight'] for task in tasks]

    dic_output = defaultdict(list)
    for _ in trange(n_gen, desc='Generating molecules'):
        set_all_seed(seed)
        res = ea.optimize(
            myAlgorithm,
            seed=seed,
            verbose=verbose,
            drawing=0,
            outputMsg=verbose,
            drawLog=False,
            saveFlag=False,
            prophet=None,
        )
        res['seed'] = seed
        seed += 1

        for k in [
            'seed',
        ]:
            dic_output[k].append(res[k])

        select_i = select_mol(res, pseudo_weights)
        latent_vec = res['Vars'][select_i]
        smi = base_model.frag_decoder.sample(
            input_noise=torch.from_numpy(latent_vec[None, ...]).float().to(device),
            output_smi=True,
        )['smi'][0]
        if smi is None:
            dic_output['smi'].append('Failed to generate a valid molecule.')
        else:
            dic_output['smi'].append(smi)
        pred_props = res['ObjV'][select_i]
        for task, pred_prop in zip(tasks, pred_props):
            dic_output['predicted ' + task['name']].append(pred_prop)

    df_output = pd.DataFrame(dic_output)
    df_output.to_csv(f'{generation_path}/generated_mols.csv', index=False)

    return df_output






















