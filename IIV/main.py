import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from _iiv_model import Net
from wavnet_dense import WaveNetDense
from iiv_miner import IIVMiner
from custom_dataset import IIVDataset
from visualize import show_vis
from train import initiate_iiv_train, initiate_train


if __name__ == '__main__':
    # 1. initiate a FastTTS train
    # initiate_train()
    # 2. Initiate a iiv train
    emo_emb_dir = "ESD/emo_reps"
    idx_emo_dict = "ESD/metadata.json"
    # Show Intra- and Inter- variation vis Before train

    show_vis()
    # Af
    initiate_iiv_train(emo_emb_dir, idx_emo_dict)