import sys

import torch
from collections import defaultdict



def get_default_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def filter_requires_grad(param_list):
    return list(filter(lambda p: p.requires_grad, param_list))


def get_FRAG_dict(model):
    dic_param = defaultdict(list)
    for name, param in model.named_parameters():
        if name.startswith('frag_encoder.'):
            dic_param['frag_encoder'].append(param)
        elif name.startswith('frag_decoder.'):
            dic_param['frag_decoder'].append(param)
        elif name.startswith('aux'):
            if 'aux_proj_fp' in name:
                dic_param['aux_proj_fp'].append(param)
            elif 'aux_proj_cl' in name:
                dic_param['aux_proj_cl'].append(param)
            else:
                raise KeyError(f'Unknown param: {name}')
        else:
            raise KeyError(f'Unknown param: {name}')

    dic_param_filtered = defaultdict(list)
    dic_param_filtered.update({k: filter_requires_grad(v) for k, v in dic_param.items()})
    return dic_param_filtered



def get_PPL_params(model, loss_object, args):
    param = model.module if args.use_multi_gpu else model
    param = get_default_params(param)

    param_1 = [
        {'params': param, 'lr': args.lr},
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.Adam(param_1, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_1, args.lr_decay, last_epoch=-1, verbose=args.is_main_proc)

    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            'params': [param_1]}


def get_FRAG_params(model, loss_object, args):
    param = model.module if args.use_multi_gpu else model
    param = get_FRAG_dict(param)

    loss_obj_param = loss_object.module if args.use_multi_gpu_for_loss_object else loss_object
    loss_obj_param = get_default_params(loss_obj_param)

    param_1 = [
        {
            'params':
                param['frag_decoder'] +
                param['frag_encoder'] +
                param['aux_proj_cl'] +
                param['aux_proj_fp'] +
                loss_obj_param,
            'lr': args.lr,
        },
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.Adam(param_1, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_1, args.lr_decay, last_epoch=-1, verbose=args.is_main_proc)
    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            'params': [param_1]}








