from utils.iiv_utils import mean_embedding_per_emo, get_emb_labels, get_anchors
from pytorch_metric_learning import distances
import os
from config.ESD.constants import num_emo
import numpy as np

def get_inter_class_distance(embs, labels):
    """
    mean_embs,  (embs_n, dim)
    labels      (embs_n, )
    """
    inter_class_dict = {}
    distance = distances.CosineSimilarity()
    mean_embs, labs = mean_embedding_per_emo(embs, labels)

    labels_list = list(range(len(labels)))
    lab_indx_pairs = [(a, b) for idx, a in enumerate(labels_list) for b in labels_list[idx + 1:]]

    for indx1, indx2 in lab_indx_pairs:
        inter_class_dict[labels[indx1] + "_" + labels[indx2]] = distance(mean_embs[indx1], mean_embs[indx2])
    return inter_class_dict


def get_intra_class_distance(embs, emo_labels, grp_labels):
    intra_class_distance = {}
    for emo in emo_labels:
        emo_mask = emo_labels == emo
        emo_mean_embs = embs[emo_mask]
        emo_grp_labels = grp_labels[emo_mask]
        grp_distances = get_inter_class_distance(emo_mean_embs, emo_grp_labels)
        intra_class_distance[emo] = grp_distances
    print(intra_class_distance)


def generate_emo_grp_repr_iiv(embs, emo_labels, grp_labels, save_dir):
    """
    generate_emo_grp_repr_iiv by mean of the corresponding embs
    Args:
        embs ():
        emo_labels ():
        grp_labels ():

    Returns:
    """
    emo_represents, labs = mean_embedding_per_emo(embs, emo_labels)
    for i, lab in enumerate(labs):
        iiv_f = os.path.join(save_dir, "{}_mean.npy".format(num_emo[str(int(lab))],
                                                               lab))
        np.save(iiv_f, emo_represents[i].detach().cpu().numpy())

    for emo in emo_labels:
        emo_mask = emo_labels == emo
        emo_mean_embs = embs[emo_mask]
        emo_grp_labels = grp_labels[emo_mask]
        grp_represents, labs = mean_embedding_per_emo(emo_mean_embs, emo_grp_labels)
        for i, lab in enumerate(labs):
            iiv_f = os.path.join(save_dir, "{}_{}_mean.npy".format(num_emo[str(int(emo))],
                                                                      lab))
            np.save(iiv_f, grp_represents[i].detach().cpu().numpy())


if __name__ == '__main__':
    base_dir = "/home/rosen/project/FastSpeech2/ESD"
    meta_data_f = base_dir + "/metadata_new.json"
    iiv_dir = base_dir + "/iiv_reps"

    embs, emo_labels, grp_labels = get_emb_labels(meta_data_f,
                                                  iiv_dir)
    # Show inter-class distance (set min)
    #inter_class_distance = get_inter_class_distance(embs, emo_labels)

    # show intra-class distance
    #intra_class_distance = get_intra_class_distance(embs, emo_labels, grp_labels)
    #print(intra_class_distance)

    # Select representative iiv embeddings
    iiv_repr_dir = "/Users/luoxuan/Project/FastSpeech2/evaluation/iiv_repr"
    generate_emo_grp_repr_iiv(
        embs,
        emo_labels,
        grp_labels,
        iiv_repr_dir
    )