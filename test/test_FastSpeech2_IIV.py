import pytest
import torch

from IIV.wavnet_dense import WaveNetDense
from IIV.iiv_miner import IIVMiner
from pytorch_metric_learning import distances
import numpy as np
from model.fastspeech2 import FastSpeech2

def test_fastspeech2():
    inputs = dict(
        text=torch.randint(0, 10, (2, 4)),
        text_lengths=torch.tensor([4, 1], dtype=torch.long),
        speech=torch.randn(2, 3, 5),
        speech_lengths=torch.tensor([3, 1], dtype=torch.long),
    )

    model = FastSpeech2()
    optimizer = ?

    loss, _ = model(**inputs)
    loss.back
    torch.manual_seed(0)
    batch = 36
    emo_nums = 4
    group_nums = 3

    all_batch = 300
    ref_embeddings = torch.randn(batch, 8)
    ref_emos = torch.randint(0, emo_nums, (batch,))
    ref_groups = torch.randint(0, group_nums, (batch,))

    all_embeddings = torch.randn(all_batch, 8)
    all_emos = torch.randint(0, emo_nums, (all_batch,))
    all_groups = torch.randint(0, group_nums, (all_batch,))

    iivminer = IIVMiner(
        embeddings=all_embeddings,
        emos=all_emos,
        groups=all_groups,
        inter_margin=0.2,
        intra_margin=0.2,
        inter_distance=distances.CosineSimilarity(),
        intra_distance=distances.CosineSimilarity(),
        inter_type_of_triplets="semihard",
        intra_type_of_triplets="semihard",
    )

    inter_indices_tuple, intra_indices_tuple_dict = iivminer.mine(
        ref_embeddings,
        ref_emos,
        ref_groups
    )
    assert len(inter_indices_tuple) == 3
    assert len(intra_indices_tuple_dict.keys()) == emo_nums
    #assert len(intra_indices_tuple_dict.values()[0]) == 3
