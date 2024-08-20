# -*- coding: utf-8 -*-
from .bn_adapt import BNAdapt
from .conjugate_pl import ConjugatePL
from .cotta import CoTTA
from .eata import EATA
from .memo import MEMO
from .no_adaptation import NoAdaptation
from .note import NOTE
from .sar import SAR
from .shot import SHOT
from .t3a import T3A
from .tent import TENT
from .ttt import TTT
from .ttt_plus_plus import TTTPlusPlus
from .dyn import DYN
from .vida import ViDA
from .rotta import Rotta
from .datta import DATTA
from .dyn_cbn import DYN_CBN
from .dyn_sbn import DYN_SBN
from .dyn_sbn_tbn import DYN_SBN_TBN
from .dyn_tbn_cbn import DYN_TBN_CBN
from .roid import ROID
from .deyo import DEYO
def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "tent": TENT,
        "bn_adapt": BNAdapt,
        "memo": MEMO,
        "shot": SHOT,
        "t3a": T3A,
        "ttt": TTT,
        "ttt_plus_plus": TTTPlusPlus,
        "note": NOTE,
        "sar": SAR,
        "conjugate_pl": ConjugatePL,
        "cotta": CoTTA,
        "eata": EATA,
        "dyn": DYN,
        "vida": ViDA,
        "rotta": Rotta,
        "datta": DATTA,
        "dyn_cbn": DYN_CBN,
        "dyn_sbn": DYN_SBN,
        "dyn_sbn_tbn": DYN_SBN_TBN,
        "dyn_tbn_cbn": DYN_TBN_CBN,
        "roid": ROID,   
        "deyo": DEYO,
    }[adaptation_name]
