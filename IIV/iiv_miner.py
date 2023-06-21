import torch
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import numpy as np

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

        if self.choose_achor:
            # inter-triplet (emb: anchors, ref_emb: pos_neg_samples)
            inter_indices_tuple = self.inter_miner(
                self.emo_mean_embedding, self.emo_mean_labels,
                ref_emb, ref_inter_labels
            )  # tuple(3, sample_num)
        # intra-triplet
        intra_indices_tuple_dict = {}
        for emo in self.emos:
            emo_mask = ref_inter_labels == emo
            if torch.any(emo_mask):
                ref_grp_embs = ref_emb[emo_mask]
                ref_intra_labs = ref_intra_labels[emo_mask]
                intra_indices_tuple = self.intra_miner(self.label_triplet_dict[emo][0], self.label_triplet_dict[emo][1],
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


def get_emb_labels(data_path):
    pass

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