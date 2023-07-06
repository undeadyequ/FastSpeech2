import json
import torch
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import numpy as np
import os
from utils.iiv_utils import mean_embedding_per_emo, mean_embedding_per_grp, get_anchors

class IIVMiner:
    """

    """
    def __init__(self,
                 data_path=None,
                 inter_margin=0.2,
                 intra_margin=0.1,
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
        #if data_path is not None:
        #    embeddings, emos, groups = get_emb_labels(data_path)
        #self.emo_mean_embedding, self.emo_mean_labels = mean_embedding_per_emo(embeddings, emos) # (emo_n, emb_dim), (emo_n,)
        #self.label_triplet_dict = mean_embedding_per_grp(embeddings, emos, groups)
        #self.emos = self.label_triplet_dict.keys()
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
        emos = torch.unique(ref_emb)
        intra_indices_tuple_dict = {}

        if self.choose_achor:
            # Choose anchor
            anchors, anchor_labs = mean_embedding_per_emo(ref_emb, ref_inter_labels)
            # inter-triplet (emb: anchors, ref_emb: pos_neg_samples)
            inter_indices_tuple = self.inter_miner(
                anchors, anchor_labs,
                ref_emb, ref_inter_labels
            )  # tuple(3, sample_num)
            # intra-triplet
            for emo in emos:
                emo_mask = ref_inter_labels == emo
                if torch.any(emo_mask):
                    ref_grp_embs = ref_emb[emo_mask]
                    ref_intra_labs = ref_intra_labels[emo_mask]
                    grp_anchors, grp_anchor_labs = mean_embedding_per_emo(ref_grp_embs, ref_intra_labs)
                    # Assign anchors instead of random selection
                    intra_indices_tuple = self.intra_miner(grp_anchors, grp_anchor_labs,
                                    ref_grp_embs, ref_intra_labs)       # tuple(3, sample_num)
                    intra_indices_tuple_dict[emo] = intra_indices_tuple # dict(emo_num, tuple(3, sample_num))
        else:
            inter_indices_tuple = self.inter_miner(
                ref_emb, ref_inter_labels
            )  # tuple(3, sample_num)

            for emo in emos:
                emo_mask = ref_inter_labels == emo
                if torch.any(emo_mask):
                    ref_grp_embs = ref_emb[emo_mask]
                    ref_intra_labs = ref_intra_labels[emo_mask]
                    intra_indices_tuple = self.intra_miner(ref_grp_embs, ref_intra_labs)  # tuple(3, sample_num)
                    intra_indices_tuple_dict[emo] = intra_indices_tuple  # dict(emo_num, tuple(3, sample_num))
        return inter_indices_tuple, intra_indices_tuple_dict