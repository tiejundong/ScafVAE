import sys

import torch
from collections import defaultdict
from diffusers.optimization import *


def get_default_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def filter_requires_grad(param_list):
    return list(filter(lambda p: p.requires_grad, param_list))


def get_xxx_dict(model):
    dic_param = defaultdict(list)
    for name, param in model.named_parameters():
        if 'pocket_sorter' in name:
            dic_param['pocket_sorter'].append(param)
        elif 'main_module' in name:
            dic_param['main_module'].append(param)
        else:
            raise KeyError(f'Unknown param: {name}')

    dic_param_filtered = defaultdict(list)
    dic_param_filtered.update({k: filter_requires_grad(v) for k, v in dic_param.items()})
    return dic_param_filtered


def get_PPLPredictor_params(model, args):
    param = get_default_params(model)

    param_1 = [
        {
            'params':
                param,
            'lr': args.lr,
        },
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.AdamW(param_1, lr=args.lr)  # Adam
    scheduler_1 = get_cosine_schedule_with_warmup(
        optimizer=optimizer_1,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.len_train_loader * args.n_epoch // args.gradient_accumulation_steps,
        num_cycles=args.n_epoch // 200,  # 0.5 for 1 to 0, e.g., args.n_epoch // 100 -> (1 to 0 to 1 every 100 epochs)
    )
    # scheduler_1 = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer=optimizer_1,
    #     num_warmup_steps=args.lr_warmup_steps,
    #     num_training_steps=args.len_train_loader * args.n_epoch // args.gradient_accumulation_steps,
    #     num_cycles=args.n_epoch // 50,
    # )


    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            'params': [param_1]}


def get_ScafVAE_params(model, args):
    param = get_default_params(model)

    param_1 = [
        {
            'params':
                param,
            'lr': args.lr,
        },
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.AdamW(param_1, lr=args.lr)  # Adam
    scheduler_1 = get_cosine_schedule_with_warmup(
        optimizer=optimizer_1,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.len_train_loader * args.n_epoch // args.gradient_accumulation_steps,
        num_cycles=args.n_epoch // 200,  # 0.5 for 1 to 0, e.g., args.n_epoch // 100 -> (1 to 0 to 1 every 100 epochs)
    )
    # scheduler_1 = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer=optimizer_1,
    #     num_warmup_steps=args.lr_warmup_steps,
    #     num_training_steps=args.len_train_loader * args.n_epoch // args.gradient_accumulation_steps,
    #     num_cycles=args.n_epoch // 50,
    # )


    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            'params': [param_1]}











