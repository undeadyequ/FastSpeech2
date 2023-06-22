import pytest
import torch
import numpy as np
from IIV.clustering import get_kmeans_label

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


