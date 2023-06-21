import pytest
import torch

from IIV.wavnet_dense import WaveNetDense

@pytest.mark.parametrize("need_weightedAverate", [False])
@pytest.mark.parametrize("need_glb_aver", [False])
def test_wavenet_dense(
        need_weightedAverate,
        need_glb_aver
):
    model = WaveNetDense(
        cdim=16,
        odim=8,
        need_weightedAverate=need_weightedAverate,
        need_glb_aver=need_glb_aver
    )

    batch = 2
    seq = 5
    channel = 16
    odim = 8
    wav2net_layer_num = 10

    if need_weightedAverate:
        inputs = dict(
            x=torch.rand(batch, wav2net_layer_num, channel, seq))
    else:
        inputs = dict(
            x=torch.rand(batch, channel, seq),
        )
    x = model(**inputs)
    assert x.shape == (batch, odim)