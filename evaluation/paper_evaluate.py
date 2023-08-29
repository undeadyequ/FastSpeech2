import os

import numpy as np
import torch
from scipy.io.wavfile import write
from audio.psd_extract import extract_opensmile
from IIV.visualize import show_iiv_distance, show_psd_contrbdim_distr, draw_contours_graph, draw_attention_contour, \
    show_confusion_table, show_emb_emo_grp_distr, show_att_emo_grp_distr, show_att_emo_grp_distr_sub
from evaluation.condition_value import get_emo_grp_iiv
from config.ESD.constants import emo_contrdim

def scatter_iiv_distr(iiv_dir, emo_dict):
    pass


def scatter_iiv_cdims(iiv_type,
                      text_type,
                      model_version,
                      syn_dir=None,
                      psd_dir=None,
                      sr=16000):
    # get iiv_embs
    if iiv_type == "mean":
        iiv_dir = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/iiv_reps"
    else:
        iiv_dir = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/iiv_reps"

    _, iiv_embs = get_emo_grp_iiv(iiv_dir)
    iiv_embs = {}
    texts = []
    model = torch.load(model_version)

    # synthesize speech by model given iiv, text
    if not os.path.isdir(syn_dir):
        syn_dir = ""
        for emo, grp_iiv in iiv_embs.items():
            for grp, iiv in iiv_embs[emo].items():
                for i, text in enumerate(texts):
                    speech = model(text, iiv)
                    syn_speech_f = os.path.join(syn_dir, "{}_{}_{}.wav".format(emo, grp, i))
                    write(syn_speech_f, sr, speech)

    # generate psd (opensmile)
    if psd_dir is None:
        extract_opensmile(syn_dir, psd_dir)

    # generate emo_group_psdembed for visualization
    emo_group_psdembed = {}
    for emo, grp_iiv in iiv_embs.items():
        emo_group_psdembed[emo] = {}
        for grp, iiv in iiv_embs[emo].items():
            emo_group_psdembed[emo][grp] = []
            for i, text in enumerate(texts):
                psd_emb_f = os.path.join(syn_dir, "{}_{}_{}.npy".format(emo, grp, i))
                psd_emb = np.load(psd_emb_f)
                emo_group_psdembed[emo][grp].append(psd_emb)
                emo_group_psdembed[emo][grp] = np.stack(emo_group_psdembed[emo][grp], axis=0)
    # draw picture
    show_psd_contrbdim_distr(emo_group_psdembed, emo_contrdim, col_n=2,
                             out_png="res/psd_contrbdim_distr.pdf")


def coutour_pitch_iiv(iiv_type,
                      text_type,
                      model_version
                      ):
    # get pitch line
    pass


def coutour_iiv_cross_attention(iiv_type, att_text_version, model_version):
    # get cross attention score
    pass

def scatter_iiv_cross_att(iiv_type, att_text_version, model_version):
    if iiv_type == "":
        iiv_dir = ""
    if att_text_version == "":
        iiv_texts = ""

    iiv_repr_embedding = {}  # dict(emo_n, dict(grp_n,  ))
    emo_att_dict = {}  # dict(emo_n, dict(grp_n, (text_n, text_len)))
    emo_att_stat_dict = {} # # dict(emo_n, dict(grp_n, (text_n, 2)))
    model = read_model(model_version)
    for emo, grp_repr_emb in iiv_repr_embedding.items():
        emo_att_dict[emo] = {}
        for grp, repr_emb in grp_repr_emb.items():
            mel_pred, att_scores = inference(iiv_texts, iiv_repr_emb)
            emo_att_dict[emo][grp] = att_scores

    # compute stats
    for emo, grp_atts in emo_att_dict.items():
        emo_att_stat_dict[emo] = {}
        for grp, atts in grp_atts.items():
            if not isinstance(atts, np.ndarray):
                atts = np.array(atts)
            att_mean = np.mean(atts, axis=1)
            att_std = np.std(atts, axis=1)
            emo_att_stat_dict[emo][grp] = [att_mean, att_std]

    # scatter the att_sd_mean in one subplot
    scatter_att_one_png = "scatter_att_distr.png"
    show_att_emo_grp_distr(emo_att_stat_dict, scatter_att_one_png)

    # scatter the att_sd_mean in 5 (=emo_num) subplot
    scatter_att_multi_png = "scatter_att_multi.png"
    show_att_emo_grp_distr_sub(scatter_att_multi_png, scatter_att_multi_png)

if __name__ == '__main__':
    output_dir = "/home/rosen/project/FastSpeech2/output/ckpt/ESD"
    model_version = "250500.pth.tar"
    model_f = os.path.join(output_dir, model_version)

    if True:
        scatter_iiv_cdims(
            iiv_type="mean",
            text_type="text50",
            model_version=model_f
        )
    if False:
        coutour_iiv_cross_attention(
            iiv_type="mean",
            att_text_version="text50",
            model_version=model_f
        )