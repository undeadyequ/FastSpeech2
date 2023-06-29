import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet, CrossAttention
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2_IIV(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2_IIV, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)

        # style
        self.style_emb = model_config["style_emb"]["exist"]
        if self.style_emb:
            if model_config["style_emb"]["fusion_style"] != "add":
                self.style_lr = nn.Linear(model_config["style_emb"]["dims"],
                                          model_config["transformer"]["encoder_hidden"])
                self.concatenator = CrossAttention(model_config["CrossAttention"])
            else:
                self.concatenator = None

        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        style_emb=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        #speech=None,
    ):
        """

        Args:
            speakers:
            texts:
            src_lens:
            max_src_len:
            mels:
            speech:    Used when style transfer    (only inference)
            style_emb: Used when style controlling (both training and testing)
            mel_lens:
            max_mel_len:
            p_targets:
            e_targets:
            d_targets:
            p_control:
            e_control:
            d_control:

        Returns:

        """

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.style_emb:
            #if speech is not None:
            #    style_emb = self.iiv_embeder(speech)
            if self.concatenator is not None:
                style_emb = self.style_lr(style_emb)
                output = self.concatenator(output, style_emb)
            else:
                output = output + style_emb.unsqueeze(1).expand(-1, max_src_len, -1)
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )