import pytest
import torch

from preprocess_ESD import prepare_grp_id
import numpy as np

colors = {
    "Neutral": 0,
    "Sad": 1,
    "Angry": 2,
    "Surprise": 3}

makers = {
    "0": "o",
    "1": "v",
    "2": "+",
    "3": "s"
}



def test_prepare_grp_id():
    out_dir = "/home/rosen/project/FG-transformer-TTS/test/test_data"
    out_dir = "/home/rosen/project/FG-transformer-TTS/ESD"

    prepare_grp_id(
        data_dir=out_dir,
        emo_clusterN=emo_clusterN
    )
