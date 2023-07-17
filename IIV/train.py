import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from IIV.wavnet_dense import WaveNetDense
from IIV.wavnet_dense import Net
from IIV.iiv_miner import IIVMiner
from IIV.custom_dataset import IIVDataset
### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from IIV.collate_fn import CommonCollateFn
from pathlib import Path
from config.ESD.constants import num_emo

### Hyperparamter

device = torch.device("cuda")

Use_intra = True

def train(model, loss_func_inter, loss_func_intra, mining_func, device, train_loader, optimizer, epoch, inter_weight):
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

        inter_tuple, grp_index_tuples, intra_mask_dict = mining_func.mine(embeddings, emo, grp)
        loss_inter = loss_func_inter(embeddings, emo, inter_tuple)


        if Use_intra:
            loss_intras = {}
            intra_sampleNums = []
            for emo, grp_tuple in grp_index_tuples.items():
                intra_sampleNums.append(len(grp_tuple[0]))
                emo_embeddings = embeddings[intra_mask_dict[emo]]
                grp_label = grp[intra_mask_dict[emo]]
                loss_intras[emo] = loss_func_intra(emo_embeddings,
                                             grp_label,
                                             grp_tuple)
            loss_intra_means = [l for l in loss_intras.values()]
            emos = [emo for emo in loss_intras.keys()]
            if len(loss_intra_means) != 0:
                loss_intra = sum(loss_intra_means) / len(loss_intra_means)
            else:
                loss_intra = 0
            loss = inter_weight * loss_inter + (1 - inter_weight) * loss_intra
        else:
            loss = loss_inter
            loss_intra = 0
            intra_sampleNums = [0]
            loss_intra_means = [0, 0, 0, 0, 0]
            emos = [0, 0, 0, 0, 0]

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {:.2f}, inter:{:.2f}, intra:{:.2f}-> {} : {}, "
                "Number of inter- and intra-triplets = {}, {}".format(
                    epoch,
                    batch_idx,
                    loss,
                    loss_inter,
                    loss_intra,
                    ",".join([num_emo[int(emo)] for emo in emos]),
                    ",".join([str(float(i))[:4] for i in loss_intra_means]),
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




def initiate_iiv_train(emo_emb_dir,
                       psd_emb_dir,
                       idx_emo_dict,
                       model="conv2d",
                       distance="cosine",
                       lossType="TripletMarginLoss",
                       choose_anchor=True,
                       inter_type_of_triplets="all",
                       intra_type_of_triplets="all",
                       inter_margin=0.2,
                       intra_margin=0.2,
                       inter_weight=0.8,
                       batch_size=512,
                       learning_rate=0.01,
                       threshold_loss_min=0,
                       model_f="best_iiv_model.pt",
                       num_epochs=10
                       ):
    """
    Train intra- and inter-variation embeddings by emotion and group labels
    Args:
        emo_emb_dir ():
        idx_emo_dict ():

    Returns:
    """
    # dataset
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
    if model == "conv2d":
        model = Net().to(device)
    else:
        model = WaveNetDense(cdim=768, odim=768).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    # pytorch-metric-learning stuff ###
    if distance == "cosine":
        distance = distances.CosineSimilarity()
    else:
        raise IOError("{} is not supported".format(distance))

    reducer = reducers.ThresholdReducer(low=threshold_loss_min)

    if lossType == "TripletMarginLoss":
        loss_func_inter = losses.TripletMarginLoss(margin=inter_margin, distance=distance, reducer=reducer)
        loss_func_intra = losses.TripletMarginLoss(margin=intra_margin, distance=distance, reducer=reducer)
    else:
        raise IOError("{} is not supported".format(lossType))

    emo_embs = [os.path.join(emo_emb_dir, indx + ".npy") for indx in dataset_train.idx_list]
    emo_mean_embs_tensor = []

    for emb_f in emo_embs:
        embs = np.load(emb_f)
        ave_embs = np.mean(embs, axis=0)
        emo_mean_embs_tensor.append(ave_embs)

    mining_func = IIVMiner(
        inter_margin=inter_margin,
        intra_margin=intra_margin,
        inter_distance=distances.CosineSimilarity(),
        intra_distance=distances.CosineSimilarity(),
        inter_type_of_triplets=inter_type_of_triplets,
        intra_type_of_triplets=intra_type_of_triplets,
        choose_anchor=choose_anchor,
    )

    # Start training
    best_acc = 0
    acc = 0
    for epoch in range(1, num_epochs + 1):
        train(model, loss_func_inter, loss_func_intra, mining_func, device, train_loader, optimizer, epoch, inter_weight)
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
    idx_emo_dict = base_dir + "ESD/metadata_22322.json"
    iivmodel = "conv2d"

    # Show Intra- and Inter- variation vis after train
    mean_anchor = False
    inter_margin = 0.2
    intra_margin = 0.2
    inter_weight = 0.5
    batch_size = 512
    model_f = "iiv_anchor{}_{}_{}_w{}_net.pt".format(mean_anchor,
                                                 str(inter_margin).replace(".", ""),
                                                 str(intra_margin).replace(".", ""),
                                                 str(inter_weight).replace(".", ""),
                                                 )
    if True:
        initiate_iiv_train(emo_emb_dir, psd_emb_dir, idx_emo_dict,
                           iivmodel,
                           choose_anchor=mean_anchor,
                           inter_type_of_triplets="semihard",
                           intra_type_of_triplets="semihard",
                           inter_margin=inter_margin,
                           intra_margin=intra_margin,
                           inter_weight=inter_weight,
                           batch_size=batch_size,
                           model_f=model_f
                           )
    if True:
        # get trained iiv embed give best model
        best_model = os.path.join(base_dir, "IIV/{}".format(model_f))
        ref_embs = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/{}".format(model_f.split(".")[0])
        get_trained_iiv_emb(emo_emb_dir, psd_emb_dir, idx_emo_dict, ref_embs, best_model)