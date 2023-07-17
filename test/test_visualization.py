import numpy as np
import pytest
from IIV.visualize import show_vis_test, show_emb_emo_grp_distr, show_iiv_distance, show_confusion_table, \
    show_psd_contrbdim_distr, draw_contours_graph, draw_attention_contour, show_emb_grp_distr
import json

colors = {
    "Neutral": "k",
    "Sad": "b",
    "Angry": "r",
    "Surprise": "y",
    "Happy": "g"
}


makers = {
    "0": "o",
    "1": "v",
    "2": "s",
    "3": "+"}


def test_visualize():
    batch = 16
    feature_dims = 5

    random = False

    psd_wavnet = "wavnet"

    #test_data_dir = "/home/rosen/project/FG-transformer-TTS/test/test_data/"
    test_data_dir = "/Users/rosen/project/FastSpeech2/preprocessed_data/ESD/iiv_reps"

    if random:
        speech_feature_emb = np.random.rand(batch, feature_dims)
        speech_emotion = np.random.randint(0, 2, (batch, 1))

    metadata_new_json = test_data_dir + "metadata_new.json"
    psd_list = []
    emo_list = []
    grp_list = []

    with open(metadata_new_json, 'r') as f:
        meta_dict = json.load(f)
        for id, v_dict in meta_dict.items():
            emo = v_dict["emotion"]
            grp = v_dict["group"]

            if psd_wavnet == "psd":
                psd_reps_dir = test_data_dir + "psd_reps"
                psd_reps_path = psd_reps_dir + "/" + id + ".npy"
                psd = np.load(psd_reps_path)
            else:
                psd_reps_dir = test_data_dir + "emo_reps"
                psd_reps_path = psd_reps_dir + "/" + id + ".npy"
                psd_temp = np.load(psd_reps_path)
                psd = np.mean(psd_temp, axis=0)
            psd_list.append(psd)
            emo_list.append(colors[emo])
            grp_list.append(makers[grp])
    show_vis_test(
        test_predicted_metrics=psd_list[:3000],
        emo_list=emo_list[:3000],
        grp_list=grp_list[:3000]
    )


def test_show_iiv_distance():
    iiv_dir = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/iiv_reps"
    emo_dict_f = "/home/rosen/project/FastSpeech2/ESD/metadata_new.json"
    show_iiv_distance(
        iiv_dir,
        emo_dict_f,
        "out_1.png"
    )


def test_show_embed_scatter(

):
    wav_n = 100
    feature_dims = 5
    emotion_n = 5
    group_n = 3
    color_keys = list(colors.keys())
    maker_keys = list(makers.keys())

    speech_feature_emb = np.random.rand(wav_n, feature_dims)
    emotions_index = np.random.randint(0, emotion_n, (wav_n, ))
    emotions_lab = np.array([color_keys[ind] for ind in emotions_index])

    groups_index = np.random.randint(0, group_n, (wav_n, ))
    groups_lab = np.array([maker_keys[ind] for ind in groups_index])
    show_emb_emo_grp_distr(speech_feature_emb, emotions_lab, groups_lab, out_png="output.png")



def test_show_emb_grp_distr():
    wav_n = 100
    feature_dims = 5
    group_n = 3
    speech_feature_emb = np.random.rand(wav_n, feature_dims)
    groups_index = np.random.randint(0, group_n, (wav_n, ))
    groups_index = np.array([str(grp) for grp in groups_index])

    show_emb_grp_distr(speech_feature_emb, groups_index, "out_cluster.png")

def test_draw_contours_graph():
    contour_len = 100
    emo_grp_pitch = {
        "Neutral": {
            "0": np.random.rand(contour_len,),
            "1": np.random.rand(contour_len,)
        },
        "Sad": {
            "0": np.random.rand(contour_len,),
            "1": np.random.rand(contour_len,)
        },
        "Angry": {
            "0": np.random.rand(contour_len,),
            "1": np.random.rand(contour_len,)
        },
        "Surprise": {
            "0": np.random.rand(contour_len,),
            "1": np.random.rand(contour_len,)
        },
        "Happy": {
            "0": np.random.rand(contour_len,),
            "1": np.random.rand(contour_len,)
        }
    }
    draw_contours_graph(emo_grp_pitch,
                        col_n=3,
                        out_png="res/emo_grp_pitch.pdf",)


def test_show_psd_contrbdim_distr():
    text_nums = 50
    psd_dim = 62
    emo_group_psdembed = {
        "Neutral": {
            "0": np.random.rand(text_nums, psd_dim),
            "1": np.random.rand(text_nums, psd_dim)
        },
        "Sad": {
            "0": np.random.rand(text_nums, psd_dim),
            "1": np.random.rand(text_nums, psd_dim)
        },
        "Angry": {
            "0": np.random.rand(text_nums, psd_dim),
            "1": np.random.rand(text_nums, psd_dim)
        },
        "Surprise": {
            "0": np.random.rand(text_nums, psd_dim),
            "1": np.random.rand(text_nums, psd_dim)
        },
        "Happy": {
            "0": np.random.rand(text_nums, psd_dim),
            "1": np.random.rand(text_nums, psd_dim)
        }
    }

    emo_contrdim = {
        "Neutral": (6, 7),
        "Sad": (6, 7),
        "Angry": (6, 51),
        "Surprise": (6, 7),
        "Happy": (6, 51),
    }

    show_psd_contrbdim_distr(emo_group_psdembed,
                             emo_contrdim,
                             col_n=2,
                             out_png="res/psd_contrbdim_distr.pdf"
                             )


def test_draw_attention_contour():
    phoneme_len = 4
    iiv_att_score = {
        "Neutral": {
            "0": np.random.rand(phoneme_len, ),
            "1": np.random.rand(phoneme_len, )
        },
        "Sad": {
            "0": np.random.rand(phoneme_len, ),
            "1": np.random.rand(phoneme_len, )
        },
        "Angry": {
            "0": np.random.rand(phoneme_len, ),
            "1": np.random.rand(phoneme_len, )
        },
        "Surprise": {
            "0": np.random.rand(phoneme_len, ),
            "1": np.random.rand(phoneme_len, )
        },
        "Happy": {
            "0": np.random.rand(phoneme_len, ),
            "1": np.random.rand(phoneme_len, )
        }
    }
    text = "YO AR HE AR."
    draw_attention_contour(
        iiv_att_score,
        text,
        out_png="res/iiv_att_score.pdf",
    )


def test_draw_mel_graph():
    pass


def test_show_confusion_table():
    actual = np.random.binomial(5, .9, size=1000)
    predicted = np.random.binomial(5, .9, size=1000)
    show_confusion_table(actual, predicted, "res/confusion_matrix.pdf")