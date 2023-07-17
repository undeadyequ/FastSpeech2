"""
1. show_iiv_distance: Show emo and group distribution by scatter plot on Wav2Net, OpenSmile and IIV embeddings
    - Also show OpenSmile embeddings for check
2. show_psd_contrbdim_distr: Show scatter plot where X,Y are the most distributable dims conditioned on different cluster IIV per emotion, with subplots nums same as emotion number.
2. draw_contours_graph: Show Pitch contours of synthesized speech conditioned by both inter-, and intra-emotion, with subplots nums same as emotion number.
4. draw_mel_graph?: Show Mel-spectrogram of synthesized speech conditioned by intra-emotion of certain emotion
5. draw_confusion_graph: Show confusion matrix of conditioned inter-emotion

"""
import json
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import collections
from sklearn import metrics
import librosa
from config.ESD.constants import num_emo

plt.rc('font', family='serif')
fontsize=12

colors = {
    "Neutral": "k",
    "Sad": "b",
    "Angry": "r",
    "Surprise": "y",
    "Happy": "g"
}


makers = {
    "0": ".",
    "1": "_",
    "2": "|",
    "3": "3"}

emo_contrdim = {
    "Neutral": (6, 7),
    "Sad": (6, 7),
    "Angry": (6, 51),
    "Surprise": (6, 7),
    "Happy": (6, 51),
}

def show_iiv_distance(
        emb_dir: str,
        idx_emo_dict_f: str,
        out_img: str = "output.png",
        max_point = 3000,
        fig_size = 20
):
    """

    Args:
        emb_dir ():
        grp_dir ():

    Returns:

    """

    with open(idx_emo_dict_f, "r") as f:
        meta_data_dir = json.load(f)

    emo_embs = []
    emo_labels = []
    grp_labels = []

    for emo_emb_f in os.listdir(emb_dir):
        id = emo_emb_f.split(".")[0]
        emo_lab = meta_data_dir[id]["emotion"]
        grp_lab = meta_data_dir[id]["group"]
        emo_emb_abs_f = os.path.join(emb_dir, emo_emb_f)
        emo_emb = np.load(emo_emb_abs_f)
        if len(emo_emb.shape) == 2:
            if emo_emb.shape[0] == 1:
                emo_emb = emo_emb.squeeze(0)
            elif emo_emb.shape[1] == 1:
                emo_emb = emo_emb.squeeze(1)
            else:
                emo_emb = emo_emb.mean(axis=0)

        emo_embs.append(emo_emb)
        emo_labels.append(emo_lab)
        grp_labels.append(grp_lab)

    emo_embs = np.array(emo_embs)
    emo_labels = np.array(emo_labels)
    grp_labels = np.array(grp_labels)

    show_emb_emo_grp_distr(emo_embs[:max_point], emo_labels[:max_point], grp_labels[:max_point], out_img, fig_size)


def show_psd_contrbdim_distr(emo_group_psdembed,
                             emo_contrdim,
                             fig_size=8,
                             col_n=3,
                             out_png="psd_contrbdim_distr.pdf"):
    emo_n = len(emo_group_psdembed.keys())
    row_n = int(emo_n / col_n) + 1
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    print(row_n, col_n)
    fig, ax = plt.subplots(row_n, col_n, figsize=(fig_size, fig_size))

    for i, emo in enumerate(emo_group_psdembed.keys()):
        legend_names = []
        scatters = []
        r = int(i / col_n)
        c = int(i % col_n)
        for g in emo_group_psdembed[emo].keys():
            intra_psd_emb = emo_group_psdembed[emo][g]
            x = intra_psd_emb[:, emo_contrdim[emo][0]]
            y = intra_psd_emb[:, emo_contrdim[emo][1]]
            # each scatter group rendering in subplot
            scatter = ax[r, c].scatter(x, y, marker=makers[g])
            scatters.append(scatter)
            legend_names.append("G{}".format(g))
        # subplot attribute
        ax[r, c].legend(
            scatters,
            legend_names,
            loc="lower left",
            markerscale=2,
            fontsize=8,
            bbox_to_anchor=(0.85, 0)
        )
        ax[r, c].set_title(emo)
    fig.savefig(out_png, dpi=300)


def draw_contours_graph(
        emo_grp_pitch,
        fig_size=6,
        col_n=3,
        out_png="prm_ang_distr.pdf",
):
    """
    emo_grp_pitch:
    {"angry": {"grp1": [0.1, 0.2, 0.5, ...],
               "grp2": [0.1, 0.2, 0.5, ...]
               },
    "happy": {"grp1": [0.1, 0.2, 0.5, ...],
               "grp2": [0.1, 0.2, 0.5, ...]
               }
    }
    """
    emo_n = len(emo_grp_pitch.keys())
    row_n = int(emo_n / col_n) + 1
    fig, ax = plt.subplots(row_n, col_n, figsize=(fig_size, fig_size))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i, emo in enumerate(emo_grp_pitch.keys()):
        legend_names = []
        plots = []
        r = int(i / col_n)
        c = int(i % col_n)
        for g in emo_grp_pitch[emo].keys():
            pitch_contour = emo_grp_pitch[emo][g]
            x_frame = range(len(pitch_contour))
            y_value = pitch_contour
            plot = ax[r, c].plot(x_frame, y_value, c)
            legend_names.append("G{}".format(g))
            plots.append(plot)

        # subplot attribute
        ax[r, c].legend(
            plots,
            legend_names,
            loc="lower left",
            markerscale=2,
            fontsize=8,
            bbox_to_anchor=(0.85, 0)
        )
        ax[r, c].set_title(emo)
        #ax[r, c].set_xticks(x_frame)
        ax[r, c].set_xlabel('Frame')
        ax[r, c].set_ylabel('Frequency (Hz)')

    plt.savefig(out_png, dpi=300)


def draw_attention_contour(
        iiv_att_score,
        text,
        fig_size=6,
        col_n=3,
        out_png="iiv_att_score.pdf",
):
    emo_n = len(iiv_att_score.keys())
    row_n = int(emo_n / col_n) + 1
    fig, ax = plt.subplots(row_n, col_n, figsize=(fig_size, fig_size))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i, emo in enumerate(iiv_att_score.keys()):
        legend_names = []
        plots = []
        r = int(i / col_n)
        c = int(i % col_n)
        for g in iiv_att_score[emo].keys():
            att_contour = iiv_att_score[emo][g]
            x_frame = range(len(att_contour))
            y_value = att_contour
            plot = ax[r, c].plot(x_frame, y_value, c)
            legend_names.append("Group{}".format(g))
            plots.append(plot)

        # subplot attribute
        ax[r, c].legend(
            plots,
            legend_names,
            #markerscale=2,
            #fontsize=8,
            #bbox_to_anchor=(0.85, 0)
        )
        ax[r, c].set_title(emo)
        ax[r, c].set_xticks(x_frame)
        ax[r, c].set_xticklabels(text.split(" "), rotation=60, fontsize=12)
        ax[r, c].set_ylabel('Attention score')

    plt.savefig(out_png, dpi=300)


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


def show_confusion_table(actual,
                         predicted,
                         out_png="psd_contrbdim_distr.pdf"):

    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    display_labels = [num_emo[str(i)] for i in range(0, len(emo_contrdim.keys()))]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    cm_display.plot()
    plt.savefig(out_png, dpi=300)



######### util ######
def show_emb_emo_grp_distr(embeddings,
                           emo_list,
                           grp_list,
                           out_png="output.png",
                           fig_size=8
                           ):
    """
    add emotion and grp agenda
    reference:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html

    Args:
        embeddings: (emb_num, dim_num)
        emo_list: (emb_num,)  exp: ("happy", )
        grp_list: (emb_num,)       ("1", )
    Returns:
    """
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    emo_sets = list(set(emo_list))
    grp_sets = list(set(grp_list))
    emo_nums = len(emo_sets)
    tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(embeddings)

    scatters = []
    legend_names = []
    for emo in emo_sets:
        cur_emo = emo_list == emo
        cur_emo_embeds = tSNE_metrics[cur_emo]
        cur_emo_grps = grp_list[cur_emo]
        color = colors[emo]
        for grp in grp_sets:
            cur_emo_grp = cur_emo_grps == grp
            cur_emo_grp_embeds = cur_emo_embeds[cur_emo_grp]
            cur_emo_grp_embeds_n = len(cur_emo_grp_embeds)
            marker = makers[grp]
            if cur_emo_grp_embeds_n != 0:
                scatter = ax.scatter(cur_emo_grp_embeds[:, 0], cur_emo_grp_embeds[:, 1], c=color,
                                     marker=marker, s=12)
                legend_names.append(emo + "_" + grp + "(Num:" + str(cur_emo_grp_embeds_n) + ")")
                scatters.append(scatter)
    # show
    ax.legend(scatters,
              legend_names,
              loc="lower left", title="Total Num:{}".format(len(emo_list)), markerscale=3, fontsize=12,
              bbox_to_anchor=(0.85, 0))
    # show annotation text
    #fig.text(0.5, 0.02, x_lab, ha='center', fontsize=fontsize)
    #fig.text(0.02, 0.5, y_lab, va='center', rotation='vertical', fontsize=fontsize)
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


if __name__ == '__main__':
    iiv_dir = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/iiv_reps"
    ssl_dir = "/home/rosen/project/FastSpeech2/ESD/emo_reps"
    psd_dir = "/home/rosen/project/FastSpeech2/ESD/psd_reps"
    emo_dict_f = "/home/rosen/project/FastSpeech2/ESD/metadata_new.json"
    maxPoint = -1

    if False:
        show_iiv_distance(
            psd_dir,
            emo_dict_f,
            "{}_emb_{}.png".format("psd", maxPoint),
            maxPoint
        )
    if True:
        for name, emb_dir in zip(["iiv", "ssl", "psd"], [iiv_dir, ssl_dir, psd_dir]):
            show_iiv_distance(
                emb_dir,
                emo_dict_f,
                "exp/{}_emb_{}.png".format(name, maxPoint),
                maxPoint,
                20
            )
            print("finished_{}".format(name))