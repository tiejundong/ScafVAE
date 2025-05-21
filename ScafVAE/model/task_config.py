from ScafVAE.model.main_layers import *
from ScafVAE.utils.dataset_utils import *
from ScafVAE.model.param_setting import *


dic_global_config = {
    'model_obj': {
        'ppl': PPLPredictor,
        'scaf': ScafVAEBase,
    },
    'dataset_obj': {
        'ppl': PPLDataset,  # PPLDataset DummyDataset
        'scaf': ScafDataset,
    },
    'collate_fn_obj': {
        'ppl': collate_ligand,  # collate_pdbbind collate_dummy
        'scaf': collate_ligand,  # collate_random_walk
    },
    'get_param': {
        'ppl': get_PPLPredictor_params,
        'scaf': get_ScafVAE_params,
    },
}






