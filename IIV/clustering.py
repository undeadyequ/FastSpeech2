from sklearn.cluster import KMeans
import numpy as np
import torch
import json
from scipy.stats import f_oneway


def choose_optimal_k(speech_feature_emb, clusterN_minMax, de_rate=0.15):
    """
    Choose optimal k by the elbow turning with decrease rate threshold
    Args:
        speech_feature_emb:
        clusterN_minMax:
        de_rate:

    Returns:
        k_sumSqr,
        de_rate_list,
        i: optimal k
    """
    k_sumSqr = {}
    groups_ids = []
    for k in range(clusterN_minMax[0], clusterN_minMax[1] + 1):
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

    de_rate_list = [(k_sumSqr[k] - k_sumSqr[k + 1]) / k_sumSqr[k]
                    for k in range(clusterN_minMax[0], clusterN_minMax[1] + 1)
                    if k < clusterN_minMax[1]]
    for i, v in enumerate(de_rate_list):
        if v < de_rate:
            return k_sumSqr, de_rate_list, clusterN_minMax[0] + i - 1


def select_contribute_dim(speech_feature_emb, group_id, dimension_name=None, emo_contri_dimN=None):

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
    psddim_fpValue_reject = {}  # (dim_n, (3, ))
    for dim, fp_value in psddim_fpValue.items():
        if fp_value[1] < 0.05:
            psddim_fpValue_reject[dim] = (psddim_fpValue[dim][0], psddim_fpValue[dim][1], dimension_name[dim])
    # Sort by p-value
    psddim_fpValue_reject_sort = sorted(psddim_fpValue_reject.items(), key=lambda item: item[1][0], reverse=True)
    print("Rejected dimension:{}".
          format(len(psddim_fpValue_reject_sort)))
    #print(*psddim_fpValue_reject_sort, sep="\n")

    # Select the most contributable dims of each cluster
    # list(top_dim_n, tuple(3,))

    if False:
        psddim_fpValue_reject_sort_fvalue = [value[1][0] for value in psddim_fpValue_reject_sort]
        top_contribute_dim_n = 1
        DEC_RATE = 0.9
        for i, v in enumerate(psddim_fpValue_reject_sort_fvalue):
            if i < len(psddim_fpValue_reject_sort_fvalue) - 1:
                decreate_rate = (v - psddim_fpValue_reject_sort_fvalue[i + 1]) / v
                if decreate_rate > DEC_RATE:
                    top_contribute_dim_n = i + 1
                    break
            else:
                top_contribute_dim_n = len(psddim_fpValue_reject_sort_fvalue)
        print("top_contribute_dim_n is {}".format(top_contribute_dim_n))
    psddim_fpValue_reject_sort_topContrb = psddim_fpValue_reject_sort[:emo_contri_dimN]

    # Got the mean value of contributable dims to each cluster (The high mean dim is assigned as name to this cluster )
    # list(cluster_n, dict{top_dim_n: value})  <- one value should be compoarably bigger than other for each cluster


    cls_contribDim_value = {}
    for cls, embs in cls_dim_clusters.items():
        cls_contribDim_value[cls] = {}
        for contrb, _ in psddim_fpValue_reject_sort_topContrb:
            dim_values = embs[:, contrb]
            dim_values_means = np.mean(dim_values)
            cls_contribDim_value[cls][contrb] = dim_values_means
    return psddim_fpValue_reject_sort_topContrb, cls_contribDim_value

def get_kmeans_label(
        speech_feature_emb,
        n_clusters = 4,
        random_state = 0,
        n_init=30
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
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    ).fit(speech_feature_emb)
    group_id = kmeans.labels_
    return group_id

def anova_test(each_psddim_all_clusters):
    return f_oneway(*each_psddim_all_clusters)