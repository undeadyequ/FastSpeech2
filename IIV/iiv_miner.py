import json
import torch
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import numpy as np
import os

class IIVMiner:
    """

    """
    def __init__(self,
                 embeddings,
                 emos,
                 groups,
                 data_path=None,
                 inter_margin=0.2,
                 intra_margin=0.2,
                 inter_distance=distances.CosineSimilarity(),
                 intra_distance=distances.CosineSimilarity(),
                 inter_type_of_triplets="semihard",
                 intra_type_of_triplets="semihard",
                 choose_anchor=False,
                 **kwargs):
        self.inter_miner = miners.TripletMarginMiner(margin=inter_margin,
                                                    distance=inter_distance,
                                                    type_of_triplets=inter_type_of_triplets,
                                                    **kwargs
                                                    )

        self.intra_miner = miners.TripletMarginMiner(margin=intra_margin,
                                                     distance=intra_distance,
                                                     type_of_triplets=intra_type_of_triplets,
                                                     **kwargs
                                                     )
        if data_path is not None:
            embeddings, emos, groups = get_emb_labels(data_path)
        self.emo_mean_embedding, self.emo_mean_labels = mean_embedding_per_emo(embeddings, emos) # (emo_n, emb_dim), (emo_n,)
        self.label_triplet_dict = mean_embedding_per_grp(embeddings, emos, groups)
        self.emos = self.label_triplet_dict.keys()
        self.choose_achor = choose_anchor

    def mine(self, ref_emb, ref_inter_labels, ref_intra_labels):
        """
        Min inter- and intra- triplet samples
        Args:
            ref_emb:            Tensor: (batch, emb_dim)
            ref_inter_labels:   Tensor: (batch,)
            ref_intra_labels:   Tensor: (batch,)

        Returns:
            inter_indices_tuple: tuple(3, sample_num)
            intra_indices_tuple_dict: dict(emo_n, tuple(Tensor: (grp_n, emb_dim), Tensor: (grp_n,)))
        """
        intra_indices_tuple_dict = {}

        if self.choose_achor:
            # Choose anchor
            anchors, anchor_labs = get_anchors(ref_emb, ref_inter_labels)
            # inter-triplet (emb: anchors, ref_emb: pos_neg_samples)
            inter_indices_tuple = self.inter_miner(
                anchors, anchor_labs,
                ref_emb, ref_inter_labels
            )  # tuple(3, sample_num)
            # intra-triplet
            for emo in self.emos:
                emo_mask = ref_inter_labels == emo
                if torch.any(emo_mask):
                    ref_grp_embs = ref_emb[emo_mask]
                    ref_intra_labs = ref_intra_labels[emo_mask]
                    grp_anchors, grp_anchor_labs = get_anchors(ref_grp_embs, ref_intra_labs)
                    intra_indices_tuple = self.intra_miner(grp_anchors, grp_anchor_labs,
                                    ref_grp_embs, ref_intra_labs)       # tuple(3, sample_num)
                    intra_indices_tuple_dict[emo] = intra_indices_tuple # dict(emo_num, tuple(3, sample_num))
        else:
            inter_indices_tuple = self.inter_miner(
                ref_emb, ref_inter_labels
            )  # tuple(3, sample_num)

            for emo in self.emos:
                emo_mask = ref_inter_labels == emo
                if torch.any(emo_mask):
                    ref_grp_embs = ref_emb[emo_mask]
                    ref_intra_labs = ref_intra_labels[emo_mask]
                    intra_indices_tuple = self.intra_miner(ref_grp_embs, ref_intra_labs)  # tuple(3, sample_num)
                    intra_indices_tuple_dict[emo] = intra_indices_tuple  # dict(emo_num, tuple(3, sample_num))
        return inter_indices_tuple, intra_indices_tuple_dict


def select_representative_iiv_embeddings(embs, emo_labels):
    return mean_embedding_per_emo(embs, emo_labels)


def get_anchors(embs, lables):
    return mean_embedding_per_emo(embs, lables)


def get_inter_class_distance(mean_embs, labels):
    """
    mean_embs,  (embs_n, dim)
    labels      (embs_n, )
    """
    inter_class_dict = {}
    distance = distances.CosineSimilarity()
    mean_embs, labs = mean_embedding_per_emo(mean_embs, labels)

    labels_list = list(range(len(labels)))
    lab_indx_pairs = [(a, b) for idx, a in enumerate(labels_list) for b in labels_list[idx + 1:]]

    for indx1, indx2 in lab_indx_pairs:
        inter_class_dict[labels[indx1] + "_" + labels[indx2]] = distance(mean_embs[indx1], mean_embs[indx2])
    return inter_class_dict


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


if __name__ == '__main__':
    meta_data_f = ""
    iiv_dir = ""
    embs, emo_labels, grp_labels = get_emb_labels(meta_data_f, iiv_dir)

    # Show inter-class distance
    inter_class_distance = get_inter_class_distance(embs, emo_labels)
    intra_class_distance = {}
    for emo in emo_labels:
        emo_mask = emo_labels == emo
        emo_mean_embs = embs[emo_mask]
        emo_grp_labels = grp_labels[emo_mask]
        grp_distances = get_inter_class_distance(emo_mean_embs, emo_grp_labels)
        inter_class_distance[emo] = grp_distances

    # Select representative iiv embeddings
    grp_represent_dict = {}
    emo_represent_dict = select_representative_iiv_embeddings(embs, emo_labels)
    for emo in emo_labels:
        emo_mask = emo_labels == emo
        emo_mean_embs = embs[emo_mask]
        emo_grp_labels = grp_labels[emo_mask]
        represent = select_representative_iiv_embeddings(emo_mean_embs, emo_grp_labels)
        grp_represent_dict[emo] = represent