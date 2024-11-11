from tqdm import tqdm
from torch.utils.data import DataLoader

from ScafVAE.model.task_config import *
from ScafVAE.utils.training_utils import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.common import *
from ScafVAE.model.main_layers import *
from ScafVAE.utils.hub import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.graph2mol import *


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ppl_predictor = PPLPredictor(None).requires_grad_(False)
        self.ppl_predictor.load_state_dict(load_PPLPredictor()['model_state_dict'])

        self.model_config = PPLPredictor_config

    def forward(self, dic_data):
        bond_ppl_ce = self.get_bond_perplexity(
            F.pad(dic_data['l_x_init'], (0, 1), 'constant', 0),
            # self.ppl_predictor.init_blank_x(dic_data['l_x_init']),
            dic_data['l_edge_label'],
            dic_data['l_mask'],
            dic_data['l_edge_all_masked_init'],
            dic_data['mask_for_edge_all_masked'],
            dic_data['scatter_idx_for_edge_all_masked'],
        )

        dic_output = dict(
            idx=dic_data['idx'],
            bond_ppl_ce=bond_ppl_ce,
            # component_idx=dic_split_result['component_idx'],
        )
        return dic_output

    @torch.no_grad()
    def get_bond_perplexity(
            self,
            x, edge_label, mask,
            l_edge_all_masked_init, mask_for_edge_all_masked, scatter_idx_for_edge_all_masked,
    ):
        if self.model_config.gt_bond_mask_chunk is None:
            single_bond_pred = self.ppl_predictor.pred(
                x[scatter_idx_for_edge_all_masked],
                l_edge_all_masked_init,
                mask=mask[scatter_idx_for_edge_all_masked],
            )
        else:
            single_bond_pred = []
            for sub_x, sub_edge, sub_mask in zip(
                    x[scatter_idx_for_edge_all_masked].split(self.model_config.gt_bond_mask_chunk, dim=0),
                    l_edge_all_masked_init.split(self.model_config.gt_bond_mask_chunk, dim=0),
                    mask[scatter_idx_for_edge_all_masked].split(self.model_config.gt_bond_mask_chunk, dim=0),
            ):
                sub_single_bond_pred = self.ppl_predictor.pred(sub_x, sub_edge, mask=sub_mask)
                single_bond_pred.append(sub_single_bond_pred)
            single_bond_pred = torch.concat(single_bond_pred, dim=0)

        ppl_ce = F.cross_entropy(
            rearrange(single_bond_pred, 'b i j d -> b d i j'),
            edge_label[scatter_idx_for_edge_all_masked],
            reduction='none',
        )

        ppl_ce = scatter_add(ppl_ce * mask_for_edge_all_masked, index=scatter_idx_for_edge_all_masked, dim=0)

        return ppl_ce



if __name__ == '__main__':
    device = 'cuda:4'
    batch_size = 128

    assert training_config.data.shuffle_input == False

    chk = torch.load(
        '/data1/dtj/ESM/experiments/ppl_max128/weights/state_5.chk',
        map_location='cpu',
    )
    args = chk['args']
    model = Model()
    model.train(False)
    model = model.to(device)

    args.som_path = '/data1/dtj/ESM/data/chembl_prepared_128/'
    data_list = [i.split('.')[0] for i in os.listdir(args.som_path)]
    args.n_batch = len(data_list)

    dataset_obj = dic_global_config['dataset_obj'][args.config]
    collate_fn = dic_global_config['collate_fn_obj'][args.config]
    test_dataset = dataset_obj('eval', data_list, args, dic_global_config)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        num_workers=2, shuffle=False,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else 0,
        collate_fn=collate_fn,
    )
    test_loader.dataset.training = False

    save_path = '/data1/dtj/ESM/data/ppl_pred'
    # delmkdir(save_path)

    for dic_data in tqdm(test_loader):
        dic_data = dic2device(dic_data, device=device)
        with torch.no_grad():
            dic_output = model(dic_data)

        for i in range(len(dic_output['idx'])):
            idx = dic_output['idx'][i]
            np.savez_compressed(f'{save_path}/{idx}.npz', bond_ppl_ce=to_numpy(dic_output['bond_ppl_ce'][i]))

    print('DONE')





