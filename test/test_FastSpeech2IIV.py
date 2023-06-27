import pytest
import torch

from IIV.wavnet_dense import WaveNetDense
from IIV.iiv_miner import IIVMiner
from pytorch_metric_learning import distances
import numpy as np
from model import FastSpeech2, ScheduledOptim, FastSpeech2Loss, FastSpeech2_IIV
import yaml


def test_fastspeech2_IIV():
    config_dir = "/home/rosen/project/FastSpeech2/config/ESD"
    train_config = config_dir + "/train.yaml"
    preprocess_config = config_dir + "/preprocess.yaml"
    model_config = config_dir + "/model.yaml"
    model_config = config_dir + "/model_fastspeechIIV.yaml"

    preprocess_config = yaml.load(
        open(preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)

    text_n = 100
    batch = 2
    restore_step = 0
    text_max_len = 31
    mel_max_len = 50
    speaker_n = 10
    style_emb_dim = 256
    speech_len = 70

    inputs = dict(
        speakers=torch.randint(0, speaker_n, (batch, )),
        texts=torch.randint(0, text_n, (batch, text_max_len)),
        src_lens=torch.randint(0, speaker_n, (batch,)),
        max_src_len=text_max_len,
        mels=torch.randn((batch, mel_max_len, 80)),
        mel_lens=torch.randint(0, mel_max_len, (batch,)),
        max_mel_len=mel_max_len,
        #speech=torch.randn((batch, speech_len)),
        style_emb=torch.randn(batch, style_emb_dim),
        p_targets=torch.randn(batch, text_max_len),
        e_targets=torch.randn(batch, text_max_len),
        d_targets=torch.randint(0, 10, (batch, text_max_len)),

    )
    inputs_value = (
        [str(l) for l in list(range(0, batch))],
        [str(l) for l in list(range(100, 100 + batch))],
        torch.randint(0, speaker_n, (batch,)),
        torch.randint(0, text_n, (batch, text_max_len)),
        torch.randint(0, speaker_n, (batch,)),
        text_max_len,
        torch.randn((batch, mel_max_len, 80)),
        torch.randint(0, mel_max_len, (batch,)),
        mel_max_len,
        torch.randn(batch, text_max_len),
        torch.randn(batch, text_max_len),
        torch.randint(0, 10, (batch, text_max_len)),
        torch.randn(batch, style_emb_dim),
    )
    model = FastSpeech2_IIV(preprocess_config, model_config)
    scheduled_optim = ScheduledOptim(
        model, train_config, model_config, restore_step
    )

    output = model(*(inputs_value[2:]))
    Loss = FastSpeech2Loss(preprocess_config, model_config)
    losses = Loss(inputs_value[:-1], output)
    print(losses)