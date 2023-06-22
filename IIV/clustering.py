from sklearn.cluster import KMeans
import numpy as np
import torch
import json
from scipy.stats import f_oneway

def choose_optimal_k(speech_feature_emb, n_clusters):
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
    print("Please choose k by checking the elbow turn from {}".format(k_sumSqr))
    #choosed_k = ""
    #group_id = groups_ids[choosed_k]
    #k = ""


def select_contribute_dim(speech_feature_emb, group_id, dimension_name=None):
    cls_dim_clusters = {}  # dict(cls_n, (sampleN_incluster, sample_dims))
    psd_dims_n = len(speech_feature_emb[0])
    k = len(list(set(group_id)))
    dims = np.shape(speech_feature_emb)[1]

    contrib_dim = {}    # list(group_id_n, [(top1_id, top1_name, fValue, pValue)])
    if dimension_name is not None:
        if isinstance(dimension_name, str):
            with open(dimension_name, "r") as f:
                dimension_name = json.load(f)
        elif isinstance(dimension_name, list):
            pass
        else:
            print("dimension_name should be str or list")

    for cls in range(k):
        cls_mask = group_id == cls
        speech_feature_emb_cls = speech_feature_emb[cls_mask]
        cls_dim_clusters[cls] = speech_feature_emb_cls
    psddim_fpValue = {}     # dict(psddim_n, (f_stats, p_value))

    for d in range(dims):
        each_psddim_all_clusters = []
        for cls in range(k):
            psd_dim_cur_cluster = cls_dim_clusters[cls][:, d]
            each_psddim_all_clusters.append(psd_dim_cur_cluster)
        f_stats, p_value = anova_test(each_psddim_all_clusters)
        psddim_fpValue[d] = (f_stats, p_value)
    # print out the vairiable whose p value less than 0.05
    psddim_fpValue_reject = {}
    for dim, fp_value in psddim_fpValue.items():
        if fp_value[1] < 0.05:
            psddim_fpValue_reject[dim] = (psddim_fpValue[dim][0], psddim_fpValue[dim][1], dimension_name[dim])
    # Sort by p-value
    psddim_fpValue_reject_sort = sorted(psddim_fpValue_reject.items(), key=lambda item: item[1][1])
    print("The anova test for each dimension, all: {} and contribute value: {}".
          format(psddim_fpValue, psddim_fpValue_reject_sort))

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
    kmeans = KMeans(
        n_clusters=n_clusters
        #random_state=random_state,
        #n_init=n_init
    ).fit(speech_feature_emb)
    group_id = kmeans.labels_
    return group_id

def anova_test(each_psddim_all_clusters):
    return f_oneway(*each_psddim_all_clusters)