import numpy as np
from audio.tools import inv_mel_spec
import audio as Audio
import yaml
import torch

mel_f = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD_bk/mel/0011-mel-0011_000001.npy"
mel_f1 = "/home/rosen/project/FastSpeech2/preprocessed_data/LJSpeech/mel/LJSpeech-mel-LJ001-0001.npy"

mel = torch.from_numpy(np.load(mel_f)).T.float()

config_f = "/home/rosen/project/FastSpeech2/config/ESD/preprocess_iiv.yaml"
config = yaml.load(open(config_f, "r"), Loader=yaml.FullLoader)
stft = Audio.stft.TacotronSTFT(
    config["preprocessing"]["stft"]["filter_length"],
    config["preprocessing"]["stft"]["hop_length"],
    config["preprocessing"]["stft"]["win_length"],
    config["preprocessing"]["mel"]["n_mel_channels"],
    config["preprocessing"]["audio"]["sampling_rate"],
    config["preprocessing"]["mel"]["mel_fmin"],
    config["preprocessing"]["mel"]["mel_fmax"],
)

inv_mel_spec(mel, "a_badsr.wav", stft)

#sft = librosa.feature.inverse.mel_to_stft(mel_np, sr=16000)
#y = librosa.griffinlim(sft)
#sf.write("a.wav", y, samplerate=16000)