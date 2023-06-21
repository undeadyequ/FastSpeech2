import os
import argparse
import json
import time

import numpy
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import librosa
import soundfile as sf
from g2p_en import G2p
from wav2vec2.wrapper import MinimalClassifier

import sys
sys.path.insert(0, '/home/rosen/project/FG-transformer-TTS/waveglow/tacotron2')
from layers import TacotronSTFT

import opensmile
from IIV.clustering import get_kmeans_label
import shutil
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=False, default="/mnt/sdb2/ESD/")
parser.add_argument('--outputdir', type=str, required=False, default="ESD")
parser.add_argument('--emo_model_dir', type=str, default=None)
parser.add_argument('--filter_length', type=int, default=1024)
parser.add_argument('--hop_length', type=int, default=256)
parser.add_argument('--win_length', type=int, default=1024)
parser.add_argument('--mel_fmin', type=float, default=0)
parser.add_argument('--mel_fmax', type=float, default=8000)
parser.add_argument('--make_test_set', action='store_true')


en_male, en_female = ['0011', '0012', '0013', '0014', '0020'], ['0015', '0016', '0017', '0018', '0019']
en = en_male + en_female

##? Only choose english

def prepare_esd():
    args = parser.parse_args()
    stft = TacotronSTFT(filter_length=args.filter_length,
                        hop_length=args.hop_length,
                        win_length=args.win_length,
                        sampling_rate=22050,
                        mel_fmin=args.mel_fmin, mel_fmax=args.mel_fmax).cuda()
    if args.emo_model_dir:
        emo_model = MinimalClassifier.load_from_checkpoint(args.emo_model_dir,
                                                           strict=False).cuda()
    else:
        emo_model = MinimalClassifier().cuda()
    emo_model.freeze()
    emo_model.eval()
    g2p = G2p()

    mel_dir = os.path.join(args.outputdir, 'mels')
    Path(mel_dir).mkdir(parents=True, exist_ok=True)
    emo_reps_dir = os.path.join(args.outputdir, 'emo_reps')
    Path(emo_reps_dir).mkdir(parents=True, exist_ok=True)
    raw_dir = os.path.join(args.outputdir, '16k_wav')
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    metadata = dict()
    root = Path(args.datadir)
    for text_file in root.rglob('*.txt'):
        if text_file.stem in en:
            encoding = 'utf-16'
            if text_file.stem in ['0011', '0020', '0015']:
                encoding = 'utf-8'
            if text_file.stem in ['0016', '0017']:
                encoding = 'latin-1'
            with open(text_file, 'r', encoding=encoding) as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    if len(line) == 3:
                        name, text, emotion = line
                        metadata[name] = {
                            'text': text,
                            'emotion': emotion
                        }

    if not args.make_test_set:
        run_set = [p for p in root.rglob('*.wav') if 'test' not in str(p) and p.stem.split("_")[0] in en]
    else:
        run_set = [p for p in root.rglob('*.wav') if 'test' in str(p) and p.stem.split("_")[0] in en]
    for audio_name in tqdm(run_set):
        audio, sr = librosa.load(audio_name, sr=None)
        audio_name = audio_name.stem
        if audio_name not in metadata:
            print(audio_name)
            continue
        length = float(len(audio)) / sr
        text = metadata[audio_name]['text']

        melspec = librosa.resample(audio, sr, 22050)
        melspec = np.clip(melspec, -1, 1)
        melspec = torch.cuda.FloatTensor(melspec).unsqueeze(0)
        melspec = stft.mel_spectrogram(melspec).squeeze(0).cpu().numpy()
        _wav = librosa.resample(audio, sr, 16000)
        _wav = np.clip(_wav, -1, 1)
        emo_reps = torch.cuda.FloatTensor(_wav).unsqueeze(0)
        emo_reps = emo_model(emo_reps).squeeze(0).cpu().numpy()

        np.save(os.path.join(mel_dir, audio_name + '.npy'), melspec)
        sf.write(os.path.join(raw_dir, audio_name + '.wav'), _wav, 16000)
        np.save(os.path.join(emo_reps_dir, audio_name + '.npy'), emo_reps)

        phonemes = g2p(text)
        metadata[audio_name].update({
            'length': length,
            'phonemes': phonemes
        })
    metadata = {k: v for k, v in metadata.items() if 'length' in v}

    with open(os.path.join(args.outputdir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

opensmile_features = opensmile.FeatureSet.GeMAPSv01a
opensmile_functional = opensmile.FeatureLevel.Functionals

def prepare_grp_id(data_dir=".", emo_clusterN=dict(), psd_extract=False):
    """
    Get kmeans group id for each emotion and append it to metadata.json, while the
    original metadata.json is saved to metadata_bk.json

    Args:
        data_dir:
        emo_clusterN:
    Returns:
        emo_group_psdID_dict
        mean_psd per emo/grp

    """
    psd_reps_dir = os.path.join(data_dir, "psd_reps")
    audio_dir = os.path.join(data_dir, "16k_wav")
    meta_data = os.path.join(data_dir, "metadata.json")
    meta_data_new = os.path.join(data_dir, "metadata_new.json")

    grp_psd_data = os.path.join(data_dir, "emo_grpPSD.npy")
    shutil.copy(meta_data, meta_data_new)

    with open(meta_data, 'r') as f:
        meta_dict = json.load(f)

    # extract and save opensmile features from each audio
    if psd_extract:
        print("smile extraction started!")
        Path(psd_reps_dir).mkdir(parents=True, exist_ok=True)
        smile = opensmile.Smile(
            feature_set=opensmile_features,
            feature_level=opensmile_functional,
        )
        dir_path = Path(audio_dir)
        for audio_f in dir_path.rglob("*.wav"):
            psd_reps = smile.process_file(audio_f)
            psd_reps = psd_reps.squeeze(0)
            audio_name = audio_f.stem
            np.save(os.path.join(psd_reps_dir, audio_name + ".npy"), psd_reps)
        print("smile extraction finished!")

    # Split audios features by emotion label
    # (I: meta.json, psd_dir. O: emo_psdId_dict)
    emo_psdId_dict = {}
    for psd_reps in os.listdir(psd_reps_dir):
        psd_reps_path = os.path.join(psd_reps_dir, psd_reps)
        psd_feature = np.load(psd_reps_path)
        id = psd_reps.split(".")[0]
        emo = meta_dict[id]["emotion"]
        if emo not in emo_psdId_dict.keys():
            emo_psdId_dict[emo] = {"psd": [], "id": []}
        emo_psdId_dict[emo]["psd"].append(psd_feature)
        emo_psdId_dict[emo]["id"].append(id)

    # get kmeans group id for each emotion
    # (I: emo_psdId_dict, emo_clusterN. O: id_gid_dict, emo_group_psdID_dict)
    id_gid_dict = {}
    emo_group_psdID_dict = {}
    for emo in emo_psdId_dict.keys():
        emo_group_psdID_dict[emo] = {}

    for emo, psd_id_dict in emo_psdId_dict.items():
        print("{} audios in {}".format(len(psd_id_dict["psd"]), emo))
        grp_ids = get_kmeans_label(psd_id_dict["psd"], emo_clusterN[emo])
        for i, id in enumerate(psd_id_dict["id"]):
            g_id = grp_ids[i]
            id_gid_dict[id] = g_id
            """
            if g_id not in emo_group_psdID_dict[emo].keys():
                g_id = str(g_id)
                emo_group_psdID_dict[emo][g_id] = {}
                emo_group_psdID_dict[emo][g_id]["psd"] = []
            emo_group_psdID_dict[emo][g_id]["psd"].append(psd_id_dict["psd"][i])
            """

    # get mean psd for each emotion_group
    """
    for emo in emo_group_psdID_dict.keys():
        for g_id in emo_group_psdID_dict[emo].keys():
            mean_psd = np.mean(emo_group_psdID_dict[emo][g_id]["psd"], axis=0)
            emo_group_psdID_dict[emo][g_id]["mean"] = mean_psd
    """

    # Aggreate group id to meta.json
    # (I: id_gid_dict, meta_dict. O: new_meata.json)
    new_meta_dict = meta_dict.copy()
    for id in new_meta_dict.keys():
        new_meta_dict[id]["group"] = str(id_gid_dict[id])
    with open(meta_data_new, 'w') as f:
        json.dump(new_meta_dict, f, indent=4)


def check_emo_group():
    """
    Verify groups by emo_groups and emo_group_psdSDmin3
    Returns:
        emo_groups = {
        "emo1": {"0": (num0, distance_sum), "1": (num1, distance_sum)},
        }

        emo_group_psdSDmin3 = {
        "emo1": {"0": ["pitch", "energy", "?"], "1": (num1, distance_sum)},
        }
    """
    emo_groups = dict()
    emo_group_psdSDmin3 = dict()

    with open(meta_data, 'r') as f:
        meta_dict = json.load(f)

    emos = np.array([])
    groups = np.array([])

    emo_uniq = numpy.unique(emos)

    for emo in emo_uniq:
        emos_mask = emos == emo
        grp_in_emo = groups[emos_mask]
        grp_uniq = numpy.unique(grp_in_emo)
        if emo not in emo_groups.keys():
            emo_groups[emo] = {}
        for grp in grp_uniq:
            grp_mask = grp_in_emo == grp
            grp_n = grp_mask.sum()
            distance_sum = ""
            psdSDmin3 = ["", "", ""]
            emo_groups[emo][grp] = (grp_n, distance_sum)
            emo_group_psdSDmin3[emo][grp] = psdSDmin3



    print("emo_groups: {}".format(emo_groups))
    print("emo_groups_psdSDmin3: {}".format(emo_group_psdSDmin3))


if __name__ == '__main__':
    #prepare_esd()
    #prepare_grp_id(data_dir=".")
    emo_clusterN = {
        "Angry": 3,
        "Surprise": 3,
        "Sad": 3,
        "Neutral": 3,
        "Happy": 3
    }
    out_dir = "/home/rosen/project/FG-transformer-TTS/ESD"
    prepare_grp_id(
        data_dir=out_dir,
        emo_clusterN=emo_clusterN
    )