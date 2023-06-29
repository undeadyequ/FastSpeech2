import numpy as np
import pytest

from IIV.visualize import show_vis_test, show_embed_scatter, show_iiv_distance
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
    iiv_dir = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD_bk/iiv_reps"
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

    show_embed_scatter(speech_feature_emb, emotions_lab, groups_lab, out_png="output.png")