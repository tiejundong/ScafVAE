import os
import sys

import argparse

from tqdm import tqdm, trange
from ray.util.multiprocessing import Pool
import pickle


from ScafVAE.utils.common import *
from ScafVAE.preprocess.bind_preprocess import process_som


def process(task):
    idx, smi, args = task
    dic_data = process_som(smi, args)
    dic_data['smi'] = smi
    pickle.dump(dic_data, open(f'{args.output_path}/{idx}.pkl', 'wb'))
    # np.savez_compressed(f'{args.output_path}/{pdb_id}.npz', **dic_data)
    return True


def try_prepare_som(intup):
    f, task = intup
    try:
        return f(task)
    except:
        return False


if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()

    # data source
    parser.add_argument('--smi_path', type=str,
                        default='/data1/dtj/ESM/data/chembl_20231201.txt', help='data path')

    parser.add_argument('--max_len_ligand', type=int,
                        default=128, help='max ligand atoms')

    # output
    parser.add_argument('--output_path', type=str,
                        default='/data1/dtj/ESM/data/chembl_prepared_128/', help='prepared path')

    args = parser.parse_args()
    print_args(args)

    print('Preparing tasks...')
    delmkdir(args.output_path)
    tasks = []
    for idx, smi in enumerate(open(args.smi_path, 'r').readlines()):
        # if os.path.exists(f'{args.output_path}/{idx}.pkl'):  # NOTE: no delmkdir(args.output_path)
        #     continue

        tasks.append((process, (idx, smi[:-1], args)))
    print(f'Task num: {len(tasks)}')

    print(f'Begin...')
    # random.shuffle(tasks)
    # for f, task in tqdm(tasks[:100]):
    #     f(task)
    # sys.exit()
    pool = Pool(36)
    succ = 0
    for r in pool.map(try_prepare_som, tasks):
        if r:
            succ += 1
    print(f'Success: {succ}/{len(tasks)}, {succ / len(tasks) * 100:.2f}%')
    print('='*20 + 'DONE' + '='*20)

