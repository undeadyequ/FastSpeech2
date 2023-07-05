import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from _iiv_model import Net
from wavnet_dense import WaveNetDense
from wavnet_dense import Net
from iiv_miner import IIVMiner
from custom_dataset import IIVDataset
from visualize import show_vis_test, show_iiv_distance
from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from collate_fn import CommonCollateFn
from pathlib import Path

### Hyperparamter
w = 0.5
intra_loss_weight_each_emo = {
    "0": 0.3,
    "1": 0.4
}
device = torch.device("cuda")
batch_size = 512
num_epochs = 100

Use_intra = True

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        emo_emb = data[1]["emo_emb"]
        emo = data[1]["emo"]
        grp = data[1]["grp"]
        emo_emb = emo_emb.permute(0, 2, 1)
        emo = torch.squeeze(emo, 1)
        grp = torch.squeeze(grp, 1)

        emo_emb, emo, grp = emo_emb.to(device), emo.to(device), grp.to(device)
        optimizer.zero_grad()
        embeddings = model(emo_emb)  # (batch, out_dim)

        inter_tuple, grp_index_tuples = mining_func.mine(embeddings, emo, grp)
        loss_inter = loss_func(embeddings, emo, inter_tuple)

        if Use_intra:
            loss_intras = {}
            intra_sampleNums = []
            for emo, grp_tuple in grp_index_tuples.items():
                intra_sampleNums.append(len(grp_tuple[0]))
                loss_intras[emo] = loss_func(embeddings, grp, grp_tuple)
            loss_intra_means = [l for l in loss_intras.values()]
            loss_intra = sum(loss_intra_means) / len(loss_intra_means)
            loss = w * loss_inter + (1 - w) * loss_intra
        else:
            loss = loss_inter
            loss_intra = 0
            intra_sampleNums = [0]

        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {:.2f}(inter:{:.2f}, intra:{:.2f}->{}, "
                "Number of inter- and intra-triplets = {}, {}".format(
                    epoch, batch_idx, loss, loss_inter, loss_intra, ",".join([str(float(i))[:4] for i in loss_intra_means]),
                    str(len(inter_tuple[0])),
                    "_".join([str(i) for i in intra_sampleNums])
                )
            )

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def tiiv(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    return accuracies


emo_num = {
    "Angry": 0,
    "Surprise": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}


def initiate_iiv_train(emo_emb_dir, psd_emb_dir, idx_emo_dict):
    """
    Train intra- and inter-variation embeddings by emotion and group labels
    Args:
        emo_emb_dir ():
        idx_emo_dict ():

    Returns:
    """

    dataset_train = IIVDataset(emo_emb_dir, psd_emb_dir, idx_emo_dict, train_test=True)
    dataset_test = IIVDataset(emo_emb_dir, psd_emb_dir, idx_emo_dict, train_test=False)  # ??? train_test ???

    collate_fn = CommonCollateFn(
        float_pad_value=0.0,
        int_pad_value=0
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                              collate_fn=collate_fn
                                              )

    # Component
    #model = WaveNetDense(cdim=768, odim=768).to(device)
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    # pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)

    all_emos = [emo_num[dict_emo_grp["emotion"]] for dict_emo_grp in dataset_train.idxv_list]
    all_grps = [int(dict_emo_grp["group"]) for dict_emo_grp in dataset_train.idxv_list]

    emo_embs = [os.path.join(emo_emb_dir, indx + ".npy") for indx in dataset_train.idx_list]
    emo_mean_embs_tensor = []

    for emb_f in emo_embs:
        embs = np.load(emb_f)
        ave_embs = np.mean(embs, axis=0)
        emo_mean_embs_tensor.append(ave_embs)

    emo_mean_embs_tensor = torch.tensor(emo_mean_embs_tensor).to(device)
    all_emos = torch.tensor(all_emos).to(device)
    all_grps = torch.tensor(all_grps).to(device)
    # Inter- and intra- mining_func
    mining_func = IIVMiner(
        emo_mean_embs_tensor,
        all_emos,
        all_grps,
        inter_margin=0.2,
        intra_margin=0.2,
        inter_distance=distances.CosineSimilarity(),
        intra_distance=distances.CosineSimilarity(),
        inter_type_of_triplets="semihard",
        intra_type_of_triplets="semihard")

    #mining_func = miners.TripletMarginMiner(
    #margin=0.2, distance=distance, type_of_triplets="semihard"
    #)

    # Start training
    best_acc = 0
    acc = 0
    model_f = "best_iiv_model.pt"
    for epoch in range(1, num_epochs + 1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        acc += 0.1
        #acc = tiiv(dataset_train, dataset_test, model, accuracy_calculator)
        if best_acc < acc:
            print("best acc: {}, save to {}".format(acc, model_f))
            best_acc = acc
            torch.save(model, model_f)

def get_trained_iiv_emb(emo_emb_dir, psd_emb_dir, idx_emo_dict, ref_embs, trained_model):
    model = torch.load(trained_model)
    dataset_train = IIVDataset(emo_emb_dir, psd_emb_dir, idx_emo_dict, train_test=True)
    collate_fn = CommonCollateFn(
        float_pad_value=0.0,
        int_pad_value=0
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, collate_fn=collate_fn
    )

    Path(ref_embs).mkdir(exist_ok=True)
    for batch_idx, data in enumerate(train_loader):
        emo_emb = data[1]["emo_emb"]
        emo_emb = emo_emb.permute(0, 2, 1)
        emo_emb = emo_emb.to(device)
        ids = data[0][0]
        iiv_embed = model(emo_emb)  # (1, 768)
        iiv_embed_n = os.path.join(ref_embs, ids + ".npy")
        np.save(iiv_embed_n, iiv_embed.detach().cpu().numpy())


if __name__ == '__main__':
    # 1. initiate a FastTTS train
    # initiate_train()
    # 2. Initiate a iiv train
    base_dir = "/home/rosen/project/FastSpeech2/"
    emo_emb_dir = base_dir + "ESD/emo_reps"
    psd_emb_dir = base_dir + "ESD/psd_reps"
    ref_embs = base_dir + "ESD/iiv_reps"
    idx_emo_dict = base_dir + "ESD/metadata_new.json"

    # Show Intra- and Inter- variation vis after train
    initiate_iiv_train(emo_emb_dir, psd_emb_dir, idx_emo_dict)

    # get trained iiv embed give best model
    best_model = os.path.join(base_dir, "IIV/best_iiv_model.pt")
    #get_trained_iiv_emb(emo_emb_dir, psd_emb_dir, idx_emo_dict, ref_embs, best_model)