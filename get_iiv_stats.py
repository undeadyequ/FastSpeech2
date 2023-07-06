from utils.iiv_utils import mean_embedding_per_emo, get_emb_labels, get_anchors
from pytorch_metric_learning import distances


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


if __name__ == '__main__':
    meta_data_f = ""
    iiv_dir = ""
    embs, emo_labels, grp_labels = get_emb_labels(meta_data_f, iiv_dir)

    # Show inter-class distance (set min)
    inter_class_distance = get_inter_class_distance(embs, emo_labels)
    intra_class_distance = {}
    for emo in emo_labels:
        emo_mask = emo_labels == emo
        emo_mean_embs = embs[emo_mask]
        emo_grp_labels = grp_labels[emo_mask]
        grp_distances = get_inter_class_distance(emo_mean_embs, emo_grp_labels)
        inter_class_distance[emo] = grp_distances
    print(intra_class_distance)

    # Select representative iiv embeddings
    grp_represent_dict = {}
    emo_represent_dict = get_anchors(embs, emo_labels)
    for emo in emo_labels:
        emo_mask = emo_labels == emo
        emo_mean_embs = embs[emo_mask]
        emo_grp_labels = grp_labels[emo_mask]
        represent = get_anchors(emo_mean_embs, emo_grp_labels)
        grp_represent_dict[emo] = represent
    print(grp_represent_dict)