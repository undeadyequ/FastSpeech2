import os

import torch
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import json

emo_num = {
    "Angry": 0,
    "Surprise": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}
class IIVDataset(Dataset):
    def __init__(self,
                 emo_emb_dir,
                 grp_emb_dir,
                 idx_emo_dict,
                 train_test="train"):
        """
        Args:
            emo_emb_dir ():  emotion embedding dir
            idx_emo_dict ():
            train_test ():
        """
        with open(idx_emo_dict, 'r') as f:
            meta_dict = json.load(f)
        self.idx_list = list(meta_dict)
        self.idxv_list = list(meta_dict.values())
        self.idx_dict = idx_emo_dict

        """
        emo_embs = [os.path.join(emo_emb_dir, indx + ".npy") for indx in self.idx_list]
        self.emo_embs_tensor = []
        self.emo_mean_embs_tensor = []

        for emb_f in emo_embs:
            embs = np.load(emb_f)
            ave_embs = np.mean(embs, axis=0)
            self.emo_embs_tensor.append(embs)
            self.emo_mean_embs_tensor.append(ave_embs)

        grps = [os.path.join(emo_emb_dir, indx + ".npy") for indx in self.idx_list]
        self.gps_embs_tensor = []
        for emb_f in grps:
            embs = np.load(emb_f)
            self.gps_embs_tensor.append(embs)
        """

        self.grp_emb_dir = grp_emb_dir
        self.emo_emb_dir = emo_emb_dir
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            uid = self.idx_list[idx]
        else:
            uid = idx
        emo_emb = np.load(os.path.join(self.emo_emb_dir, self.idx_list[idx] + ".npy"))
        #emo_emb = np.transpose(emo_emb, (1, 0))  # (dim, seqs)

        emo_lab = np.array([emo_num[self.idxv_list[idx]["emotion"]]])
        emo_lab.astype(int)

        grp_lab = np.array([int(self.idxv_list[idx]["group"])])
        #text = self.idx_dict[self.idxv_list[idx]]["text"]
        #length = self.idx_dict[self.idxv_list[idx]]["length"]
        #phoneme = self.idx_dict[self.idxv_list[idx]]["phonemes"]

        data = {
            "emo_emb": emo_emb,
            "emo": emo_lab,
            "grp": grp_lab
        }
        return uid, data