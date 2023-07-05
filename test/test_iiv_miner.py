import pytest
import torch

from IIV.wavnet_dense import WaveNetDense
from IIV.iiv_miner import IIVMiner
from pytorch_metric_learning import distances

import numpy as np
@pytest.mark.parametrize("choose_anchor", [(False)])
def test_iiv_miner(
        choose_anchor
):
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
        choose_anchor=choose_anchor
    )

    inter_indices_tuple, intra_indices_tuple_dict = iivminer.mine(
        ref_embeddings,
        ref_emos,
        ref_groups
    )
    assert len(inter_indices_tuple) == 3
    assert len(intra_indices_tuple_dict.keys()) == emo_nums
    #assert len(intra_indices_tuple_dict.values()[0]) == 3