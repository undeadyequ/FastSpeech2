import os
from config.ESD.constants import emo_num



def get_emo_grp_iiv(repr_type, iiv_repr_dir):
    emo_represent_dict = {}
    grp_represent_dict = {}
    if repr_type == "mean":
        emo_represent_dict, grp_represent_dict = read_iiv_repr(iiv_repr_dir)
    elif repr_type == "":
        pass
    else:
        print("{} is not support".format(repr_type))
    return emo_represent_dict, grp_represent_dict


def get_emo_grp_rfaudio(repr_type):
    pass


def get_object_text(text_type):
    pass


def get_subject_text(text_type):
    pass


def get_trained_model(model_ver):
    pass


def get_att_text(text_type):
    pass


def get_pitch_text(text_type):
    pass

##########
def read_iiv_repr(iiv_repr_dir):
    pass