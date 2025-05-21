import os
import torch
import ScafVAE
from ScafVAE.model.config import *



def load_PPLPredictor():
    # print('Loading PPLPredictor parameters ...')
    chk = torch.load(
        f'{root_path}/params/ppl_param.chk',
        map_location='cpu',
    )

    return chk


def load_ModelBase():
    print('Loading ScafVAE parameters ...')
    chk = torch.load(
        f'{root_path}/params/ScafVAE.chk',
        map_location='cpu',
    )

    return chk



if __name__ == '__main__':
    pass




