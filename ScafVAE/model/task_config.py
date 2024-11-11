from ScafVAE.model.main_layers import *
# from ScafVAE.model.sampler import *
from ScafVAE.model.loss import *
from ScafVAE.utils.dataset_utils import *
from ScafVAE.model.param_setting import *


dic_global_config = {
    'model_obj': {
        'PPL': PPLPredictor,
        'FRAG': ModelBase,
    },
    'loss_obj': {
        'PPL': PPLLossFunction,
        'FRAG': FragLossFunction,
    },
    'dataset_obj': {
        'PPL': PPLDataset,
        'FRAG': ComplexDataset,
    },
    'collate_fn_obj': {
        'PPL': collate_ppl,
        'FRAG': collate_ligand,
    },
    'get_param': {
        'PPL': get_PPL_params,
        'FRAG': get_FRAG_params,
    },
}






