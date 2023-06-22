from sklearn.cluster import KMeans
import numpy as np
import torch
import json
from scipy.stats import f_oneway

def get_kmeans_label(
        speech_feature_emb,
        n_clusters = 4,
        random_state = 0,
        n_init="auto"
):
    """
    Get group ids for samples of each category
    Args:
        speech_path ():
        text_path ():
        emo_lab_path ():

    Returns:
        grp_ids: {"speech_id": g_id}
    """
    if not isinstance(speech_feature_emb, np.ndarray):
        speech_feature_emb = np.array(speech_feature_emb)
    # Cluster
    if isinstance(n_clusters, int):
        kmeans = KMeans(
            n_clusters=n_clusters
            #random_state=random_state,
            #n_init=n_init
        ).fit(speech_feature_emb)
        group_id = kmeans.labels_
        k = n_clusters

    # Choose optimal k
    elif isinstance(n_clusters, tuple) and len(n_clusters) == 2:
        k_sumSqr = {}
        groups_ids = []
        for k in range(n_clusters[0], n_clusters[1] + 1):
            kmeans = KMeans(
                n_clusters=k
                # random_state=random_state,
                # n_init=n_init
            ).fit(speech_feature_emb)
            group_id = kmeans.labels_
            sum_sqr_dtc = kmeans.inertia_
            k_sumSqr[k] = sum_sqr_dtc
            groups_ids.append(group_id)
        print("please choose k by checking the elbow turn from {}".format(k_sumSqr))
        choosed_k = ""
        group_id = groups_ids[choosed_k]
        k = ""
    else:
        group_id = ""
        print("{} only accept int or tuple".format(n_clusters))

    # Investigate most contribute dimension
    psd_dim_clusters = {}
    psd_dims_n = len(speech_feature_emb[0])
    with open("", "r") as f:
        psd_dims_names = json.load(f)  # list(dims_n, )
    for cls in range(k):
        psd_dim_clusters[cls] = []
        cls_mask = group_id == cls
        speech_feature_emb_cls = speech_feature_emb[cls_mask]
        for d in range(psd_dims_n):
            psd_dim_cluster = speech_feature_emb_cls[:, d]
            psd_dim_clusters[cls].append(psd_dim_cluster)
    anova_res_each_psddim = {}
    for d in range(psd_dims_n):
        each_psddim_all_clusters = []
        for cls in range(k):
            psd_dim_cur_cluster = psd_dim_clusters[cls][:, d]
            each_psddim_all_clusters.append(psd_dim_cur_cluster)
            f_stats, p_value = anova_test(each_psddim_all_clusters)
            anova_res_each_psddim[d] = (f_stats, p_value)
    print("please check the contributing dimension by looking over {} and {}".
          format(anova_res_each_psddim, psd_dims_names))
    return group_id

def anova_test(each_psddim_all_clusters):
    return f_oneway(*each_psddim_all_clusters)