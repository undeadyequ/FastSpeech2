import pytest
import torch
import numpy as np
from IIV.clustering import get_kmeans_label, choose_optimal_k, select_contribute_dim

@pytest.mark.parametrize("n_clusters", [(2, 6)])
def test_get_kmeans_label(
        n_clusters
):
    batch = 16
    feature_dims = 1000
    speech_feature_emb = np.random.rand(batch, feature_dims)

    group_ids = get_kmeans_label(
        speech_feature_emb,
        n_clusters=n_clusters)

    assert group_ids.shape == (batch,)


def test_choose_optimal_k(
        n_clusters=(2, 4)
):
    batch = 16
    feature_dims = 1000
    speech_feature_emb = np.random.rand(batch, feature_dims)

    group_ids = choose_optimal_k(
        speech_feature_emb,
        clusterN_minMax=n_clusters)

def test_select_contribute_value(
        n_clusters=4
):
    batch = 16
    feature_dims = 1000
    speech_feature_emb = np.random.rand(batch, feature_dims)

    group_ids = get_kmeans_label(
        speech_feature_emb,
        n_clusters=n_clusters)
    dimension_name = list(range(feature_dims))
    select_contribute_dim(speech_feature_emb, group_ids, dimension_name)