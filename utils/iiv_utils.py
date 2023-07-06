import json
import torch
import numpy as np
import os

def mean_embedding_per_emo(embedding: torch.Tensor,
                           labels: torch.Tensor):
    """
    Get average embedding per emotion, used as inter anchors for inter-emotion training
    Args:
        embedding:
        labels:

    Returns:
        emo_mean_embedding: Tensor: (emo_n, emb_dim)
        emo_mean_labels: Tensor: (emo_n,)
    """
    mean_embs = []
    mean_labels = []

    label_uniqs = torch.unique(labels)
    for lab in label_uniqs:
        lab_mask = lab == labels
        embs_in_emo = embedding[lab_mask]
        embs_mean_in_emo = embs_in_emo.mean(0)
        mean_embs.append(embs_mean_in_emo)
        mean_labels.append(lab)
    return torch.stack(mean_embs), torch.stack(mean_labels)


def mean_embedding_per_grp(embedding, inter_labels, intra_labels):
    """
    Get average embedding per group for each emotion, used as intra anchors for intra-emotion training
    Args:
        embedding:
        labels:
    Returns:
        label_triplet_dict: dict(
        emo_n, tuple(Tensor: (grp_n, emb_dim), Tensor: (grp_n,))
        )
    """
    label_triplet_dict = {}
    label_uniqs = torch.unique(inter_labels)
    for lab in label_uniqs:
        lab_mask = lab == inter_labels
        embs_in_emo = embedding[lab_mask]
        sublab_in_emo = intra_labels[lab_mask]
        mean_embs, mean_labels = mean_embedding_per_emo(embs_in_emo, sublab_in_emo)
        label_triplet_dict[lab] = (mean_embs, mean_labels)
    return label_triplet_dict



def get_emb_labels(meta_data_f, iiv_dir):
    iiv_embs = []
    emos = []
    grps = []
    with open(meta_data_f, 'r') as f:
        meta_dict = json.load(f)
    for id, attr_val in meta_dict.items():
        iiv_emb_f = os.path.join(iiv_dir, id + ".npy")
        iiv_emb = np.load(iiv_emb_f)
        emo = attr_val["emotion"]
        grp = attr_val["group"]

        iiv_embs.append(iiv_emb)
        emos.append(emo)
        grps.append(grp)
    return iiv_embs, emos, grps


def get_anchors(embs, lables):
    return mean_embedding_per_emo(embs, lables)