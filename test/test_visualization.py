import numpy as np
import pytest

from IIV.visualize import show_vis
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
    test_data_dir = "/home/rosen/project/FG-transformer-TTS/ESD/"

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
    show_vis(
        test_predicted_metrics=psd_list[:3000],
        emo_list=emo_list[:3000],
        grp_list=grp_list[:3000]
    )