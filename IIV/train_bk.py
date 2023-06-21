import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import ThresholdReducer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from _iiv_model import Net
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import miners, losses


def visualize(test_predicted_metrics):
    tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(test_predicted_metrics[-1])
    plt.scatter(tSNE_metrics[:, 0], tSNE_metrics[:, 1], c=test_true_labels[-1])
    plt.colorbar()
    plt.savefig("output_wav2net.png")
    plt.show()

# 学習用関数

def select_embedding():
    pass
def kmeans (embedding, labels):
    embedding_emoi_groups, embedding_emoi_group_ids = [], []
    for l in enumerate(set(labels)):
        embedding_emoi = select_embedding(embedding, l)
        embedding_emoi_group, embedding_emoi_group_id = kmeans(embedding_emoi)
    return embedding_emoi_groups, embedding_emoi_group_ids

embedding_group_dict = {}
w = 0.7

def train(model, loss_func, mining_func, device, dataloader, optimizer, epoch):
    model.train()
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(inputs)

        #group_ids = embedding_group_dict[embeddings]

        # Custom Miner
        ## inter_tuple: (c_p, x^p_i, x^n_*), intra_tuple: (c^p,g, x^p_i, x^p_j)
        ### x^n_* should be close to c_p
        ### c_p, c^p,g is mean of all positive samples in batch and all positive in group g of group
        #inter_tuple, intra_tuple = mining_func(embeddings, labels, group_ids)
        inter_tuple = mining_func(embeddings, labels)

        loss = loss_func(embeddings, labels, inter_tuple)
        #loss1 = loss_func(embeddings, labels, inter_tuple)
        #loss2 = loss_func(embeddings, group_ids, inter_tuple)
        #loss = w * loss1 + (1 - w) * loss2
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}'.format(epoch, idx, loss, mining_func.num_triplets))
    print()


# テスト用関数
def test(model, dataloader, device, epoch):
    _predicted_metrics = []
    _true_labels = []
    model.eval()
    with torch.no_grad():
        for i, (inputs,  labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            metric = model(inputs).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            _predicted_metrics.append(metric)
            _true_labels.append(labels.detach().cpu().numpy())
    return np.concatenate(_predicted_metrics), np.concatenate(_true_labels)

# パラメーター
epochs = 50
lr = 1e-4
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
test_predicted_metrics = []
test_true_labels = []

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST(".", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(
    dataset1, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

for epoch in range(1, epochs + 1):
    print('Epoch {}/{}'.format(epoch, epochs))
    print('-' * 10)
    train(model, loss_func, miner, device, train_loader, optimizer, epoch)
    _tmp_metrics, _tmp_labels = test(model, test_loader, epoch)
    test_predicted_metrics.append(_tmp_metrics)
    test_true_labels.append(_tmp_labels)
