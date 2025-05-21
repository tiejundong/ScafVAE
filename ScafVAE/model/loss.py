import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add
from einops import rearrange, repeat, reduce
import functools

from ScafVAE.model.config import *
from ScafVAE.utils.data_utils import *
from ScafVAE.utils.common import *
from ScafVAE.utils.graph2mol import *



class LossFunction(torch.nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.dic_loss = {}
        self.dic_log = {}
        self.loss_weight = None

    def get_loss_weight(self, k, epoch=-1):
        if isinstance(self.loss_weight[k], Iterable):
            if epoch >= len(self.loss_weight[k]) or epoch == -1:
                return self.loss_weight[k][-1]
            else:
                return self.loss_weight[k][epoch]
        else:
            return self.loss_weight[k]

    def masked_mean(self, x, mask):
        return (x * mask).sum() / (mask.sum() + EPS)

    def proj_loss(self, pred, true, mask, fn):
        loss = fn(
            pred,
            true,
            reduction='none',
        )
        loss = reduce(
            loss * mask, 'b ... -> b', 'sum',
        ) / reduce(mask + EPS, 'b ... -> b', 'sum')
        loss = loss.mean()
        return loss


class PPLLossFunction(LossFunction):
    def __init__(self):
        super().__init__()
        self.loss_weight = training_config.loss_weight

    def forward(self, dic_data, epoch=1000):
        self.get_single_bond_loss(dic_data)

        grad_loss = 0
        for k in self.dic_loss.keys():
            sub_loss = self.dic_loss[k] * self.get_loss_weight(k, epoch=epoch)
            grad_loss = grad_loss + sub_loss

        # eval
        self.dic_log.update(load_dic_log(self.dic_loss))
        dic_log = {k: v.item() if is_tensor(v) else v for k, v in self.dic_log.items()}

        return grad_loss, dic_log

    def get_single_bond_loss(self, dic_data):
        mask = dic_data['mask_for_edge_single_masked'].bool()
        pred = dic_data['l_edge_single_masked_pred'][mask]
        label = dic_data['l_edge_label'][mask]

        l_single_bond_loss = F.cross_entropy(pred, label, reduction='mean')

        self.dic_loss['l_single_bond_loss'] = l_single_bond_loss


class ScafLossFunction(LossFunction):
    def __init__(self, model_config=None):
        super().__init__()
        self.loss_config = training_config.loss
        self.loss_weight = training_config.loss_weight

        self.model_config = ScafVAE_config if model_config is None else model_config

        if training_config.loss.n_cl_pool > 0:
            self.register_buffer(
                'cl_embed_1_pool',
                torch.randn(training_config.loss.n_cl_pool, model_config.cl_hidden),
            )
            self.register_buffer(
                'cl_embed_2_pool',
                torch.randn(training_config.loss.n_cl_pool, model_config.cl_hidden),
            )

    def forward(self, dic_data, epoch=1000):
        self.get_scaf_loss(dic_data)
        self.get_aa_loss(dic_data)
        self.get_kl_loss(dic_data)

        if self.model_config.with_aux:
            self.get_mix_loss(dic_data)

        main_loss = 0
        for k in self.dic_loss.keys():
            sub_loss = self.dic_loss[k] * self.get_loss_weight(k, epoch=epoch)
            main_loss = main_loss + sub_loss
        grad_loss = (main_loss,)  # update_together == False

        # eval
        self.dic_log.update(load_dic_log(self.dic_loss))
        dic_log = {k: v.item() if is_tensor(v) else v for k, v in self.dic_log.items()}

        return grad_loss, dic_log

    def get_scaf_loss(self, dic_data):
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

    def get_aa_loss(self, dic_data):
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

    def get_kl_loss(self, dic_data):
        kl_loss = self.calc_KLD(dic_data['noise_mean'], dic_data['noise_logvar'])
        kl_loss = kl_loss.mean()
        self.dic_loss['kl_loss'] = kl_loss

    def calc_KLD(self, mu, logvar):
        KLD = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD.sum(dim=-1)
        return KLD

    def get_mix_loss(self, dic_data):
        cl_embed_mask_1 = dic_data['cl_embed_mask_1']
        cl_embed_mask_2 = dic_data['cl_embed_mask_2']

        if training_config.loss.n_cl_pool > 0:
            x = torch.concat([
                cl_embed_mask_1, self.cl_embed_1_pool,
                cl_embed_mask_2, self.cl_embed_2_pool,
            ], dim=0)
        else:
            x = torch.concat([
                cl_embed_mask_1,
                cl_embed_mask_2,
            ], dim=0)
        cos_sim = get_cosine_similarity(x.unsqueeze(dim=1), x.unsqueeze(dim=0), chunk=64) / self.loss_config.cl_T
        cos_sim[torch.eye(cos_sim.shape[0], device=cos_sim.device, dtype=torch.bool)] = -get_max_tensor(cos_sim)
        B = x.shape[0] // 2
        label = torch.arange(B * 2, device=cos_sim.device)
        label[:B] = label[:B] + B
        label[B:2 * B] = label[B:2 * B] - B
        cl_loss = F.cross_entropy(
            cos_sim,
            label,
            reduction='mean',
        )
        self.dic_loss['cl_loss'] = cl_loss

        self.dic_loss['cl_embed_1_norm_penalty'] = self.get_norm_penalty(cl_embed_mask_1)
        self.dic_loss['cl_embed_2_norm_penalty'] = self.get_norm_penalty(cl_embed_mask_2)

        if training_config.loss.n_cl_pool > 0:
            accelerator = dic_data['accelerator']
            self.cl_embed_1_pool = torch.concat(
                [
                    self.cl_embed_1_pool,
                    accelerator.gather(dic_data['cl_embed_mask_1']).detach(),
                ], dim=0,
            )[-self.cl_embed_1_pool.shape[0]:]
            self.cl_embed_2_pool = torch.concat(
                [
                    self.cl_embed_2_pool,
                    accelerator.gather(dic_data['cl_embed_mask_2']).detach(),
                ], dim=0,
            )[-self.cl_embed_2_pool.shape[0]:]


        self.dic_loss['fp_loss'] = F.binary_cross_entropy_with_logits(
            dic_data['fp_embed'], dic_data['fp'],
            # pos_weight=torch.empty_like(dic_data['fp']).fill_(5.0),
            reduction='mean',
        )

        self.dic_log['fp_acc'] = (
            (dic_data['fp_embed'].sigmoid() > 0.5).long() == dic_data['fp']
        ).sum() / (dic_data['fp'].shape[0] * dic_data['fp'].shape[1])

        # with torch.no_grad():
        #     self.dic_log['fp_no_balance'] = F.binary_cross_entropy_with_logits(
        #         dic_data['fp_embed'], dic_data['fp'],
        #         reduction='mean',
        #     )

    def get_norm_penalty(self, x):
        return ((x.norm(p=2, dim=-1) - 1) ** 2).mean()

