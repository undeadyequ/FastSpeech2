import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNetDense(nn.Module):
    """
    Dense the Wav2Net2 features to a psd ordered embedding
    """
    def __init__(self,
                 cdim,
                 odim,
                 conv_chans_list=(128, 128),
                 conv_layers=2,
                 batch_norm=False,
                 kernal_size=3,
                 conv_stride=2,
                 lstm_units=128,
                 lstm_layers=1,
                 drop_rate=0.2,
                 need_weightedAverate=False,
                 need_glb_aver=False
                 ):
        """
        Args:
            cdim ():
                the channel dimension, consisted of output from each transformer layer
                it is ignored if need_weightedAverate is False.
            odim (): the nums of emotions
            conv_layers ():
            batch_norm ():
        """
        super(WaveNetDense, self).__init__()
        self.need_weightedAverage = need_weightedAverate
        self.need_glb_aver = need_glb_aver
        # need WeightedAverage
        if need_weightedAverate:
            self.weight = torch.nn.Parameter(torch.ones(cdim, 1))
            self.weight.requires_grad = True

        # pointwise conv1d
        convs = []
        padding = (kernal_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = cdim if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv1d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=kernal_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False
                ),
                #torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(drop_rate)
            ]
        self.convs = torch.nn.Sequential(*convs)

        # LSTM layer
        lstm_in_units = conv_chans_list[-1]
        self.lstm = torch.nn.LSTM(lstm_in_units, lstm_units, lstm_layers, batch_first=True)

        # Global Average (Not for training) ??? Why need this ???
        if need_glb_aver:
            self.glb_average = torch.nn.AvgPool1d(kernel_size=1)
        # Dense
        self.dense = torch.nn.Linear(lstm_units, odim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (): (batch, seq, idim, channel)
            or,
            x (): (batch, seq, idim)
        Returns:
            x : (batch, odim)
        """
        if self.need_weightedAverage:
            x = torch.mul(x, self.weight)
            x = x.squeeze(dim=2)
        #
        #x = x.unsqueeze(1)  # (b, c, l)
        x = self.convs(x)   # (b, c_o, l_o)
        x = x.transpose(1, 2)  # (b, l_o, c_o)

        #batch, l = x.size(0), x.size(1)
        #x = x.contiguous().view(batch, l, -1)
        self.lstm.flatten_parameters()
        out, hid = self.lstm(x)
        if self.need_glb_aver:
            x = self.glb_average(out)
        hid = hid[-1].squeeze(0)
        x = self.dense(hid)     # (b, l_o, c_o)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)

    def forward(self, x):
        x = torch.mean(x, dim=2)
        #x = torch.squeeze(x, dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x