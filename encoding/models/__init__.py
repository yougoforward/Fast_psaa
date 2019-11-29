from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *
from .aanet import *
from .aanet_ml import *
from .dict_aanet import *
from .topk_aanet import *
from .pydict import *
from .pydict_encnet import *
from .aanet_encnet import *
from .aanet_nopam import *
from .aanet_pam_metric import *
from .amca_asppaca import *
from .aspoc_secam import *
from .asp_oc_gap_secam import *
from .pgfnet import *
from .asp_pgfnet import *
from .asppacaca import *
from .aspoc_gsecam_net import *
from .aspoc_gsecam_du_net import *
from .fcn_8s import *
from .fcn_du import *
from .aanet_simple import *
from .amca import *
from .amca2 import *
from .aca import *
from .aca2 import *
from .aspp import *
from .aspp2 import *
from .cam import *
from .cam2 import *
from .asppcam import *
from .asppaca import *
from .amcacam import *
from .amcaaca import *
from .papnet import *
from .papnet2 import *
from .papnet3 import *
from .papnet4 import *
from .papnet5 import *
from .papnet6 import *
from .papnet7 import *
from .psp2 import *
from .papnet8 import *
from .papnet9 import *
from .psp3 import *
from .psp4 import *
from .psp5 import *
from .psp6 import *
from .psp7 import *
from .psaa import *
from .psaa2 import *
from .new_psp import *
from .psaa3 import *
from .psaa32 import *
from .psaa33 import *
from .psaa34 import *
from .psaa35 import *

from .psaa4 import *
from .psaa5 import *
from .psaa6 import *
from .psaa7 import *
from .psaa8 import *
from .psaa9 import *
from .psaa10 import *
from .psaa62 import *
from .aspoc import *
from .aspoc_psaa import *
from .new_psp3 import *
from .new_psp4 import *
from .new_psp5 import *
from .new_psp6 import *

from .aspoc_psaa_gp import *
from .psaa52 import *
from .psaa53 import *

from .new_psp2 import *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,
        'aanet': get_aanet,
        'aanet_ml': get_aanet_ml,
        'dict_aanet': get_dict_aanet,
        'topk_aanet': get_topk_aanet,
        'pydict': get_pydict,
        'pydict_encnet': get_pydict_encnet,
        'aanet_encnet': get_aanet_encnet,
        'aanet_nopam': get_aanet_nopam,
        'aanet_metric': get_aanet_metric,
        'amca_asppaca': get_amca_aspp_acanet,
        'aspoc_secam': get_aspoc_secamnet,
        'asp_oc_gap_secam': get_asp_oc_gap_secamnet,
        'pgfnet': get_pgfnet,
        'asp_pgfnet': get_asp_pgfnet,
        'asppacaca': get_asppacaca,
        'aspoc_gsecam': get_aspoc_secamnet,
        'aspoc_gsecam_du': get_aspoc_gsecam_dunet,
        'fcn_8s': get_fcn_8s,
        'fcn_du': get_fcn_du,
        'aanet_simple': get_aanet_fast,
        'amca': get_amcanet,
        'amca2': get_amca2net,
        'aca': get_acanet,
        'aca2': get_aca2net,
        'aspp': get_asppnet,
        'aspp2': get_aspp2net,
        'cam': get_camnet,
        'cam2': get_cam2net,
        'asppcam': get_asppcamnet,
        'asppaca': get_asppacanet,
        'amcaaca': get_amcacamnet,
        'amcacam': get_amcacamnet,
        'pap': get_papnet,
        'pap2': get_pap2net,
        'pap3': get_pap3net,
        'pap4': get_pap4net,
        'pap5': get_pap5net,
        'pap6': get_pap6net,
        'pap7': get_pap7net,
        'psp2': get_psp2,
        'pap8': get_pap8net,
        'psp3': get_psp3net,
        'psp4': get_psp4net,
        'psp5': get_psp5net,
        'psp6': get_psp6net,
        'psp7': get_psp7net,
        'pap9': get_pap9net,
        'psaa': get_psaanet,
        'psaa2': get_psaa2net,
        'psaa3': get_psaa3net,
        'psaa32': get_psaa32net,
        'psaa33': get_psaa33net,
        'psaa34': get_psaa34net,
        'psaa35': get_psaa35net,

        'psaa4': get_psaa4net,
        'psaa5': get_psaa5net,
        'psaa6': get_psaa6net,
        'psaa7': get_psaa7net,
        'psaa8': get_psaa8net,
        'psaa9': get_psaa9net,
        'psaa10': get_psaa10net,
        'psaa62': get_psaa62net,
        'psaa52': get_psaa52net,
        'psaa53': get_psaa53net,

        'aspoc': get_aspocnet,
        'aspoc_psaa':get_aspoc_psaanet,
        'aspoc_psaa_gp':get_aspoc_psaa_gpnet,
        'new_psp': get_new_psp,
        'new_psp2': get_new_psp2,

        'new_psp3': get_new_psp3,
        'new_psp4': get_new_psp4,
        'new_psp5': get_new_psp5net,
        'new_psp6': get_new_psp6net,

    }
    return models[name.lower()](**kwargs)
