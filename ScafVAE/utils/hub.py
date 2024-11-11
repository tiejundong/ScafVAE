import os
import torch
import ScafVAE
from ScafVAE.model.config import *


root_path = os.path.dirname(ScafVAE.__file__)


def load_PPLPredictor():
    # print('Loading PPLPredictor parameters ...')
    chk = torch.load(
        f'{root_path}/ppl_predictor_param/ppl_param.chk',
        map_location='cpu',
    )

    return chk


def load_ModelBase():
    print('Loading ScafVAE parameters ...')
    chk = torch.load(
        f'{model_param_path}/ScafVAE.chk',
        map_location='cpu',
    )

    return chk



if __name__ == '__main__':
    pass




