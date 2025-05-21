import numpy as np
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ScafVAE.utils.common import *


from collections import OrderedDict


def save_data_split(train_list, val_list, test_list, path='./'):
    delmkdir(path, remove_old=False)
    save_idx_list(train_list, f'{path}/train_list.txt')
    save_idx_list(val_list, f'{path}/val_list.txt')
    save_idx_list(test_list, f'{path}/test_list.txt')


def load_data_split(path='./', blind_training=False):
    train_list = load_idx_list(f'{path}/train_list.txt')
    val_list = load_idx_list(f'{path}/val_list.txt')
    test_list = load_idx_list(f'{path}/test_list.txt')

    if blind_training:
        train_list += val_list

    return train_list, val_list, test_list


def get_dataloader(args, train_dataset, val_dataset, collate_fn, val_batch_size=4):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else 0,
        prefetch_factor=2, collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=val_batch_size,
        num_workers=2, shuffle=False,
        persistent_workers=args.persistent_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


class CustomOptimization:
    def __init__(self, args, dic_opt):
        self.args = args
        self.optimizers = dic_opt['optimizer']
        self.schedulers = dic_opt['scheduler']

        self.N = len(self.optimizers)

    def __call__(self, accelerator, model, grad_loss, use_accelerator=True):
        if is_tensor(grad_loss):
            grad_loss = [grad_loss]

        assert isinstance(grad_loss, tuple) or isinstance(grad_loss, list)
        assert len(grad_loss) == self.N

        # if no NVLink, use local SGD
        # with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
        # local_sgd.step()

        for opt_step_i in range(self.N):
            retain_graph = True if opt_step_i < self.N - 1 else False

            if use_accelerator:
                # with torch.autograd.detect_anomaly():  # for debug
                accelerator.backward(grad_loss[opt_step_i], retain_graph=retain_graph)

                self.optimizers[opt_step_i].step()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

            else:
                grad_loss[opt_step_i].backward( retain_graph=retain_graph)
                self.optimizers[opt_step_i].step()

            self.optimizers[opt_step_i].zero_grad()

        self.update_schedulers()

    def update_schedulers(self):
        for opt_step_i in range(self.N):
            self.schedulers[opt_step_i].step()

    def save_state_dict(self):
        dic_save = {
            **{f'opt_{i}': o.state_dict() for i, o in enumerate(self.optimizers)},
            **{f'sche_{i}': s.state_dict() for i, s in enumerate(self.schedulers)},
        }
        return dic_save

    def load_state_dict(self, dic_save, args):
        for i in range(self.N):
            self.optimizers[i].load_state_dict(dic_save[f'opt_{i}'])
            self.schedulers[i].load_state_dict(dic_save[f'sche_{i}'])

            # self.schedulers[i].step(args.restart * args.len_train_loader // args.num_processes)

    def init_accelerator(self, accelerator):
        for i in range(self.N):
            self.optimizers[i] = accelerator.prepare(self.optimizers[i])
            self.schedulers[i] = accelerator.prepare(self.schedulers[i])


def weights_init(model, t=None, a=0, b=1, gain=1):
    if t is None:
        return
    assert t in ['uniform_', 'xavier_normal_', 'zeros_', 'ones_']
    for p in model.parameters():
        if t == 'uniform_':
            torch.nn.init.uniform_(p, a=a, b=b)
        elif t == 'xavier_normal_':
            if len(p.shape) > 1:
                torch.nn.init.xavier_normal_(p, gain=gain)
        elif t == 'zeros_':
            torch.nn.init.zeros_(p)
        elif t == 'ones_':
            torch.nn.init.ones_(p)


def save_params(args, my_model, opt_objects, dic_traj, save_path, exclude_param=None):
    # use unwrapped_model = accelerator.unwrap_model(model)
    model_state_dict = my_model.module.state_dict() if args.distributed_type == 'MULTI_GPU' else my_model.state_dict()

    for k, v in list(model_state_dict.items()):
        # if v.requires_grad == False:
        #     model_state_dict.pop(k)
        #     continue
        if exclude_param is not None:
            for p in exclude_param:
                if k.startswith(p):
                    model_state_dict.pop(k)
                    continue
    torch.save(
        {
            'model_state_dict': model_state_dict,
            'dic_traj': dic_traj,
            'args': args,
            **opt_objects.save_state_dict(),
         },
        save_path)


def dic2device(dic_data, device, epoch=-1):
    for k in dic_data.keys():
        if is_tensor(dic_data[k]):
            dic_data[k] = dic_data[k].to(device)
    dic_data['device'] = device
    dic_data['epoch'] = epoch

    return dic_data


def init_params(args, my_model, opt_objects, strict_param=True):
    if args.restart == 0:
        if args.start_weight is not None:
            if args.is_main_process:
                print(f'Loading params from {args.start_weight}')
            chk = torch.load(args.start_weight, map_location='cpu')

            if not strict_param:
                allow_leys = [k for k, v in my_model.named_parameters()]
                remove_keys = []
                for k in chk['model_state_dict'].keys():
                    if k not in allow_leys:
                        remove_keys.append(k)
                for k in remove_keys:
                    del chk['model_state_dict'][k]
                if len(remove_keys) > 0 and args.is_main_process:
                    print(f'strict_param is {strict_param}')
                    print(f'Ignoring {len(remove_keys)} parameters !!!')
                    if len(remove_keys) < 10:
                        print(remove_keys)

            my_model.load_state_dict(chk['model_state_dict'], strict=strict_param)
            del chk
        else:
            weights_init(my_model)

        dic_traj = {'train': defaultdict(list), 'val': defaultdict(list)}
        if args.is_main_process:
            delmkdir(args.weight_path)
            delmkdir(args.vis_path)

    else:
        if args.is_main_process:
            print(f'Loading params from {args.weight_path}/state_{args.restart}.chk')
        chk = torch.load(f'{args.weight_path}/state_{args.restart}.chk', map_location='cpu')

        my_model.load_state_dict(chk['model_state_dict'], strict=strict_param)

        opt_objects.load_state_dict(chk, args)

        dic_traj = chk['dic_traj']

        if args.is_main_process:
            print('Restart at [{}/{}], current lr: {:.2e}'.format(
                args.restart,
                args.n_epoch,
                opt_objects.optimizers[0].param_groups[0]['lr']),
            )
        del chk

    return my_model, opt_objects, dic_traj


def init_tensorboard(log_dir, log_port, restart, dic_traj, start_new_tensorboard=True, drop_head_epoch=0, sele_env=None):
    '''
    # e.g. tensorboard --port 6007 --logdir /root/tf-logs/
    # dic_traj: {'train': defaultdict(list), 'val': defaultdict(list)}
    '''
    delmkdir(log_dir)
    if start_new_tensorboard:
        print(f'Kill old tensorboards and tried to start a new one on (port:{log_port}, logdir:{log_dir})')
        os.system('ps -ef | grep tensorboard | grep -v grep | awk \'{print "kill -9 "$2}\' | sh')
        cmd = [
            '#!/bin/bash',
            # if need select env
            f'conda activate {sele_env}' if not isinstance(sele_env, type(None)) else 'echo Tensorboard: Using current env',
            f'nohup tensorboard --port {log_port} --logdir {log_dir} > /dev/null 2>&1 &'
        ]
        sh_path = f'{log_dir}/start.sh'
        open(sh_path, 'w').write('\n'.join(cmd))
        os.system(f'nohup bash -i {sh_path} > /dev/null 2>&1 &')
    else:
        print(f'Tensorboard CMD: tensorboard --port {log_port} --logdir {log_dir}')
    writer = SummaryWriter(log_dir=log_dir)
    if restart != 0:
        for epoch in range(restart):
            update_tensorboard(writer, dic_traj, epoch, drop_head_epoch=drop_head_epoch)
    return writer


def update_tensorboard(writer, dic_traj, epoch, drop_head_epoch=0, add_single=False):
    if epoch < drop_head_epoch:
        return None

    if add_single:
        for k in dic_traj.keys():
            if k.endswith('_loss'):
                folder = 'Loss'
            elif k.endswith('_metric'):
                folder = 'Metric'
            else:
                folder = 'Other'

            writer.add_scalar(f'{folder}/{k}', dic_traj[k], global_step=epoch)

    else:
        for k in dic_traj['train'].keys():
            if k.endswith('_loss'):
                folder = 'Loss'
            elif k.endswith('_metric'):
                folder = 'Metric'
            else:
                folder = 'Other'
            writer.add_scalars(f'{folder}/{k}',
                               {'Train': np.array(dic_traj['train'][k][epoch]),
                                'Val': np.array(dic_traj['val'][k][epoch])},
                               global_step=epoch)


class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)






















