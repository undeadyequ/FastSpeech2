"""
1. Show scatter plot of Wav2Net and IIV embeddings
    - Also show OpenSmile embeddings for check
2. Show Pitch/energy contours of synthesized speech conditioned by inter-emotion

3. Show Pitch/energy contours of synthesized speech conditioned by intra-emotion of certain emotion

4. Show Mel-spectrogram of synthesized speech conditioned by intra-emotion of certain emotion

5. Show cross table of conditioned intra-emotion and the most varied style of synthesized speech.
"""

import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import collections
import librosa
plt.rc('font', family='serif')
fontsize=12

def show_embed_scatter(embeddings,
                       emo_list,
                       grp_list,
                       out_png="output.png"
                       ):
    """
    Args:
        embeddings: (emb_num, dim_num)
        emo_list: (emb_num,)
        grp_list: (emb_num,)
    Returns:
    """
    fig, ax = plt.subplots()
    emo_nums = len(list(set(emo_list)))
    tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(embeddings)

    scatters = []
    for emo in range(emo_nums):
        cur_emo = emo_list == emo
        cur_emo_embeds = tSNE_metrics[cur_emo]
        cur_emo_grps = grp_list[cur_emo]
        cur_emo_grp_num = len(list(set(cur_emo_grps)))
        for grp in range(cur_emo_grp_num):
            cur_emo_grp = cur_emo_grps == grp
            cur_emo_grp_embeds = cur_emo_embeds[cur_emo_grp]
            cur_emo_grp_embeds_n = len(cur_emo_grp_embeds)
            scatter = ax.scatter(cur_emo_grp_embeds[:, 0], cur_emo_grp_embeds[:, 1], [emo] * cur_emo_grp_embeds_n,
                                 [grp] * cur_emo_grp_embeds_n
                                 )
            scatters.append(scatter)
    # show
    ax.legend(scatters, loc="lower left", title="Classes", scatterpoints=1, fontsize=8)
    plt.savefig("output.png")

    # show annotation text
    #fig.text(0.5, 0.02, x_lab, ha='center', fontsize=fontsize)
    #fig.text(0.02, 0.5, y_lab, va='center', rotation='vertical', fontsize=fontsize)
    fig.savefig(out_png, dpi=300)


def draw_mel_graph(
        inter_intra_mels
):
    """
    inter-emo_n * intra_emo_n
    inter_intra_mels:
    {"angry": {"intra_emo1": np.array(seq_n, mel_dim), "intra_emo2": np.array(seq_n, mel_dim)},
    "happy": {},
    }
    """
    # check intra_emo_max_num
    max_intra_emo_num = 0
    for emo, intra_dict in inter_intra_mels.items():
        intra_n = len(intra_dict.keys())
        if intra_n > max_intra_emo_num:
            max_intra_emo_num = intra_n
    emo_n = len(inter_intra_mels.keys())
    fig, axes = plt.subplots(emo_n, max_intra_emo_num)

    for i, emo in enumerate(list(inter_intra_mels.keys())):
        for j, intra in enumerate(list(inter_intra_mels[emo].keys())):
            mel_spec = inter_intra_mels[emo][intra]
            if isinstance(mel_spec, str):
                y, sr = librosa.load(librosa.ex(mel_spec))
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                                   fmax=8000)
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                img = librosa.display.specshow(mel_spec, x_axis='time',
                                               y_axis='mel', sr=sr,
                                               fmax=8000, ax=axes[i, j])
                fig.colorbar(img, ax=axes[i, j], format='%+2.0f dB')
                axes[i, j].set(title='Mel-frequency spectrogram')


def draw_contours_graph(
        psd_change_dict,
        psd_order,
        plot_title=None,
        psd_units=None,
        col_n=3,
        x_lab=None,
        y_lab=None,
        out_png="prm_ang_distr.pdf",
        psd_type="prm"
):
    """
    psd_change_dict:
    {"angry": {"intra_emo1": {"strength_0": [0.1, 0.2, 0.5, ...],
                              "strength_0.5": [0.3, ...],
                              "strength_1":[0.1, 0.2, ...]
                              }}}
    """
    #plt.figure(figsize=(60, 10))
    emo_n = len(list(psd_change_dict.keys()))
    psd_n = len(list(psd_change_dict.values())[0].keys())
    subplots_n = emo_n * psd_n

    row_n = int(subplots_n / col_n) + 1
    ratios = col_n / row_n
    fig, axes = plt.subplots(row_n, col_n, figsize=(10 * ratios, 10))

    #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0.3, hspace=0.4)
    if psd_type == "psd":
        plt.subplots_adjust(wspace=0.3, hspace=0.4)

    n = 0
    sample_txt = "They forcefully keep them at a black hotel"
    for i, emo in enumerate(["ang", "neu"]):
        for j, psd in enumerate(psd_order):
            bias_psdcg = psd_change_dict[emo][psd]
            bias_psdcg_ordered = collections.OrderedDict(sorted(bias_psdcg.items()))
            for bias, psdcg in bias_psdcg_ordered.items():
                r = int(n / col_n)
                c = int(n % col_n)
                x_frame = range(len(psdcg))
                y_value = psdcg
                # draw lines
                axes[r, c].plot(x_frame, y_value, label=bias)

                # set title
                if plot_title is not None:
                    subplot_title = plot_title[psd]
                else:
                    subplot_title = psd
                #subplot_title = "Conditioning on " + subplot_title.lower()
                axes[r, c].set_title(subplot_title, size=12)

                #axes[r, c].set_xticks([0, 0.5, 1])
                # set x, y label
                if psd_units is not None:
                    axes[r, c].set_ylabel(psd_units[psd])

                    # Only show xticklabels at pictures in last row.
                if r != row_n - 1 and (r + 1) * col_n + c < subplots_n:
                    plt.setp(axes[r, c].get_xticklabels(), visible=False)
                    if psd_type == "prm":
                        axes[r, c].set_xticks(x_frame)
                else:
                    if psd_type == "prm":
                        axes[r, c].set_xticks(x_frame)
                        axes[r, c].set_xticklabels(sample_txt.split(" "), rotation=60, fontsize=12)
                    elif psd_type == "psd":
                        axes[r, c].set_xlabel('Frame')
            # set legend
            legend = axes[r, c].legend(loc="upper right")
            #legend.set_alpha(None)
            n += 1

    for i in range(row_n * col_n - subplots_n):
        last_subplot_col = subplots_n % col_n
        fig.delaxes(axes[row_n - 1][last_subplot_col + i])

    if x_lab is not None:
        if psd_type == "psd":
            fig.text(0.5, 0.65, "Angry", ha='center', fontsize=fontsize)
            fig.text(0.5, 0.35, "Neutrual", ha='center', fontsize=fontsize)
        elif psd_type == "prm":
            fig.text(0.5, 0.62, "Angry", ha='center', fontsize=fontsize)
            fig.text(0.5, 0.30, "Neutrual", ha='center', fontsize=fontsize)

    if y_lab is not None:
        if psd_type == "psd":
            fig.text(0.1, 0.6, y_lab, va='center', rotation='vertical', fontsize=fontsize)
        elif psd_type == "prm":
            fig.text(0.07, 0.62, y_lab, va='center', rotation='vertical', fontsize=fontsize)
    fig.savefig(out_png, dpi=300)


def show_vis_test(embeddings,
             emo_list,
             grp_list):
    tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(embeddings)
    for x, y, c, s in zip(tSNE_metrics[:, 0], tSNE_metrics[:, 1], emo_list, grp_list):
        plt.scatter(x, y, s=12, c=c, marker=s)

    # legend of color and marker
    #
    plt.savefig("output.png")
    plt.show()


def show_iiv_distance(
        emb_dir: str,
        idx_emo_dict_f: str
):
    """

    Args:
        emb_dir ():
        grp_dir ():

    Returns:

    """

    idx_emo_dict = read_dict(idx_emo_dict_f)
    emo_embs = []
    emo_labels = []
    grp_labels = []

    for emo_emb_f in os.listdir(emb_dir):
        id = emo_emb_f.split(".")[0]
        emo_lab = idx_emo_dict[id]["emotion"]
        grp_lab = idx_emo_dict[id]["group"]
        emo_emb = np.load(emo_emb_f)

        emo_embs.append(emo_emb)
        emo_labels.append(emo_lab)
        grp_labels.append(grp_lab)

    tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(emo_embs)

    for e in list(set(emo_labels)):
        e_index = emo_labels.index(e)
        grp_e_labels = grp_labels.index(e_index)
        plt.scatter(tSNE_metrics[:, 0],
                    tSNE_metrics[:, 1],
                    c=grp_e_labels,
                    marker=e_index
                    )
        plt.colorbar()
        plt.savefig("output_wav2net.png")
        plt.show()


def read_dict(dict_f: str):
    return {}