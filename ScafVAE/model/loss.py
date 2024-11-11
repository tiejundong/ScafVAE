import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add
from einops import rearrange, repeat, reduce

from ScafVAE.model.config import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.common import *


class LossFunction(torch.nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.dic_loss = {}
        self.eval_loss = {}
        self.loss_weight = None

    def get_loss_weight(self, k, epoch=-1):
        if isinstance(self.loss_weight[k], Iterable):
            if epoch >= len(self.loss_weight[k]) or epoch == -1:
                return self.loss_weight[k][-1]
            else:
                return self.loss_weight[k][epoch]
        else:
            return self.loss_weight[k]


class PPLLossFunction(LossFunction):
    def __init__(self, args, loss_config=None, loss_weight=None):
        super(PPLLossFunction, self).__init__()
        self.loss_config = training_config.loss if loss_config is None else loss_config
        self.loss_weight = training_config.loss_weight.PPL if loss_config is None else loss_weight

    def forward(self, dic_data, epoch=1000):
        self.get_single_bond_loss(dic_data)

        main_loss = 0
        for k in self.dic_loss.keys():
            sub_loss = self.dic_loss[k] * self.get_loss_weight(k, epoch=epoch)
            main_loss = main_loss + sub_loss
        grad_loss = (main_loss,)  # update_together == False

        # eval
        self.eval_loss.update(load_eval_loss(self.dic_loss))
        eval_loss = {k: to_numpy(v) for k, v in self.eval_loss.items()}

        return grad_loss, eval_loss

    def get_single_bond_loss(self, dic_data):
        mask = dic_data['mask_for_edge_single_masked'].bool()
        pred = dic_data['l_edge_single_masked_pred'][mask]
        label = dic_data['l_edge_label'][mask]

        l_single_bond_loss = F.cross_entropy(pred, label, reduction='mean')

        self.dic_loss['l_single_bond_loss'] = l_single_bond_loss


class FragLossFunction(LossFunction):
    def __init__(self, args, loss_config=None, loss_weight=None):
        super(FragLossFunction, self).__init__()
        self.loss_config = training_config.loss if loss_config is None else loss_config
        self.loss_weight = training_config.loss_weight.FRAG if loss_config is None else loss_weight

        self.model_config = ScafVAE_config

    def forward(self, dic_data, epoch=1000):
        self.get_scaf_loss(dic_data, epoch)
        self.get_aa_loss(dic_data, epoch)
        self.get_kl_loss(dic_data, epoch)

        if self.model_config.with_aux:
            self.get_mix_loss(dic_data, epoch)

        main_loss = 0
        for k in self.dic_loss.keys():
            sub_loss = self.dic_loss[k] * self.get_loss_weight(k, epoch=epoch)
            main_loss = main_loss + sub_loss
        grad_loss = (main_loss,)  # update_together == False

        # eval
        self.eval_loss.update(load_eval_loss(self.dic_loss))
        eval_loss = {k: to_numpy(v) for k, v in self.eval_loss.items()}

        return grad_loss, eval_loss

    def get_scaf_loss(self, dic_data, epoch):
        x_mask = pad_one(dic_data['pooling_mask'])
        x_scaf_pred_loss = F.cross_entropy(
            rearrange(dic_data['x_scaf_pred'], 'n b d -> b d n'),
            dic_data['scaf_idx_bfs_with_end_token'],
            reduction='none',
        )
        x_scaf_pred_loss = (x_scaf_pred_loss * x_mask).sum(dim=-1)  # / (x_mask.sum(dim=-1) + EPS)
        x_scaf_pred_loss = x_scaf_pred_loss.mean()

        pooling_mask = dic_data['pooling_mask']
        bridge_posi = (
            dic_data['reordered_scaf_sparse_adj_bfs'] < (self.model_config.ppl_force_max_atom_with_ring ** 2)
        ).long().triu(diagonal=1).argmax(dim=-2)
        edge_scaf_pred = dic_data['edge_scaf_pred']
        edge_scaf_pred.masked_fill_(
            torch.ones_like(edge_scaf_pred).tril(diagonal=0).bool(), -get_max_tensor(edge_scaf_pred))
        edge_scaf_pred_loss = F.cross_entropy(
            rearrange(edge_scaf_pred, 'b i j -> b i j'),
            bridge_posi,
            reduction='none',
        )
        edge_scaf_pred_loss = reduce(
            edge_scaf_pred_loss[:, 2:] * pooling_mask[:, 2:], 'b j -> b', 'sum',
        )  # / (reduce(pooling_mask[:, 2:], 'b j -> b', 'sum') + EPS)
        edge_scaf_pred_loss = edge_scaf_pred_loss.mean()

        self.dic_loss['x_scaf_pred_loss'] = x_scaf_pred_loss
        self.dic_loss['edge_scaf_pred_loss'] = edge_scaf_pred_loss

    def get_aa_loss(self, dic_data, epoch):
        x_aa_pred_loss = F.cross_entropy(
            rearrange(dic_data['x_aa_pred'], 'b n d -> b d n'),
            dic_data['l_x_init'].argmax(dim=-1),
            reduction='none',
        )
        x_aa_pred_loss = (
            x_aa_pred_loss * dic_data['x_aa_masked_posi']
        ).sum(dim=-1)  # / (dic_data['x_aa_masked_posi'].sum(dim=-1) + 1e-5)
        x_aa_pred_loss = x_aa_pred_loss.mean()

        self.dic_loss['x_aa_pred_loss'] = x_aa_pred_loss

        bridge_posi = (
                dic_data['reordered_scaf_sparse_adj_bfs'] < (self.model_config.ppl_force_max_atom_with_ring ** 2)
        ).triu(diagonal=1).bool()
        edge_bond_pred = dic_data['edge_bond_pred']
        edge_bond_pred.masked_fill_(~dic_data['edge_allow_mask'].bool(), -get_max_tensor(edge_bond_pred))
        edge_bond_pred = edge_bond_pred[bridge_posi]
        edge_bond_true = dic_data['reordered_scaf_sparse_adj_bfs'][bridge_posi]

        batch_idx = rearrange(
            torch.arange(dic_data['edge_bond_pred'].shape[0], device=edge_bond_pred.device),
            'b -> b () ()',
        ) * torch.ones_like(dic_data['reordered_scaf_sparse_adj_bfs'])
        batch_idx = batch_idx[bridge_posi]
        edge_bond_pred_loss = F.cross_entropy(
            edge_bond_pred,
            edge_bond_true,
            reduction='none',
        )

        # avoid inf, some data inconsistent with data preprocessing, fix this bug in future
        tol = 1e+2
        noinf_mask = (-tol < edge_bond_pred_loss).float() * (edge_bond_pred_loss < tol ).float()
        edge_bond_pred_loss = edge_bond_pred_loss * noinf_mask

        edge_bond_pred_loss = scatter_add(edge_bond_pred_loss, index=batch_idx)
        edge_bond_pred_loss = edge_bond_pred_loss.mean()

        self.dic_loss['edge_bond_pred_loss'] = edge_bond_pred_loss

    def get_kl_loss(self, dic_data, epoch):
        kl_loss = self.calc_KLD(dic_data['noise_mean'], dic_data['noise_logvar'])
        kl_loss = kl_loss.mean()
        self.dic_loss['kl_loss'] = kl_loss

    def calc_KLD(self, mu, logvar):
        KLD = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD.sum(dim=-1)
        return KLD

    def get_mix_loss(self, dic_data, epoch):
        cl_embed_mask_1 = dic_data['cl_embed_mask_1']
        cl_embed_mask_2 = dic_data['cl_embed_mask_2']

        x = torch.concat([
            cl_embed_mask_1,
            cl_embed_mask_2,
        ], dim=0)
        sim = get_cosine_similarity(x.unsqueeze(dim=1), x.unsqueeze(dim=0)) / self.loss_config.cl_T
        diag_mask = torch.eye(sim.shape[1], device=sim.device)
        sim.masked_fill_(diag_mask.bool(), -get_max_tensor(sim))

        n_half = len(sim) // 2
        small_half = torch.arange(n_half, device=sim.device)
        label = torch.concat([small_half + n_half, small_half])

        cl_loss = F.cross_entropy(
            sim,
            label,
            reduction='mean',
        )
        self.dic_loss['cl_loss'] = cl_loss

        self.dic_loss['cl_embed_1_norm_penalty'] = self.get_norm_penalty(cl_embed_mask_1)
        self.dic_loss['cl_embed_2_norm_penalty'] = self.get_norm_penalty(cl_embed_mask_2)

        self.dic_loss['fp_loss'] = F.binary_cross_entropy_with_logits(
            dic_data['fp_embed'], dic_data['fp'],
            # pos_weight=torch.empty_like(dic_data['fp']).fill_(5.0),
            reduction='mean',
        )

        self.eval_loss['fp_acc'] = (
            (dic_data['fp_embed'].sigmoid() > 0.5).long() == dic_data['fp']
        ).sum() / (dic_data['fp'].shape[0] * dic_data['fp'].shape[1])

        # with torch.no_grad():
        #     self.eval_loss['fp_no_balance'] = F.binary_cross_entropy_with_logits(
        #         dic_data['fp_embed'], dic_data['fp'],
        #         reduction='mean',
        #     )

    def get_norm_penalty(self, x):
        return ((x.norm(p=2, dim=-1) - 1) ** 2).mean()


class TaskedLossFunction(LossFunction):
    def __init__(self, task_type, n_task_out, task_loss_weight):
        super(TaskedLossFunction, self).__init__()
        self.base_loss_obj = FragLossFunction(None)

        self.task_type = task_type
        self.n_task_out = n_task_out
        self.task_loss_weight = task_loss_weight

    def forward(self, dic_data, epoch=1000):
        grad_loss, eval_loss = self.base_loss_obj(dic_data)

        task_loss = self.get_task_loss(dic_data)
        task_loss = task_loss * self.task_loss_weight

        grad_loss = grad_loss + task_loss
        eval_loss['task_loss'] = to_numpy(task_loss)

        return grad_loss, eval_loss

    def get_task_loss(self, dic_data):
        task_pred = dic_data['task_pred']
        task_true = dic_data['task_label']

        if self.task_type == 'classification':
            if self.n_task_out == 1:
                task_pred = task_pred.squeeze(dim=-1)

                if len(task_true.shape) == 2 and task_true.shape[-1] == 1:
                    task_true = task_true.squeeze(dim=-1)
                task_true = task_true.float()
                loss = F.binary_cross_entropy_with_logits(task_pred, task_true, reduction='mean')
            else:
                task_true = task_true.long()
                if len(task_true.shape) == 2:
                    task_true = task_true.argmax(dim=-1)
                loss = F.cross_entropy(task_pred, task_true, reduction='mean')

        elif self.task_type == 'regression':
            # loss = F.smooth_l1_loss(task_pred, task_true, beta=0.1, reduction='mean')
            loss = F.mse_loss(task_pred, task_true, reduction='mean')

        else:
            raise KeyError

        return loss













