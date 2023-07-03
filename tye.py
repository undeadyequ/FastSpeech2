import numpy as np
from audio.tools import inv_mel_spec
import audio as Audio
import yaml
import torch
import librosa
import scipy
from audio.tools import get_mel_from_wav
from utils.model import vocoder_infer, get_vocoder
from scipy.io.wavfile import write

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_f = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD_bk/mel/0011-mel-0011_000001.npy"
mel_f1 = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/mel/0014-mel-0014_001606.npy"
mel_f2 = "/home/rosen/project/FastSpeech2/preprocessed_data/ESD/mel/0018-mel-0018_001204.npy"

wav_f1 = "/home/rosen/project/FastSpeech2/0014_001606.wav"
wav_f2 = "/home/rosen/project/FastSpeech2/0018_001204.wav"  # female


grd_wav = "/home/rosen/project/FastSpeech2/0014_001606.wav"

# librosa mel
y, sr = librosa.load(wav_f2, sr=16000)

if False:
    mel_libro = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, n_mels=80, hop_length=512)
    print(mel_libro)
    # Inverse mel to wav
    #inv_mel_spec(mel_libro, "a_libro.wav", stft)
    S1 = librosa.feature.inverse.mel_to_audio(mel_libro, sr=sr, n_fft=1024, hop_length=512)
    #S2 = librosa.feature.inverse.mel_to_stft(mel_libro, sr=sr, n_fft=1024)
    #y = librosa.griffinlim(S2, hop_length=512)
    #librosa.output.write_wav("a_libro.wav", y, 16000)
    librosa.output.write_wav("b_libro_ori.wav", S1, 16000)


#sft = librosa.feature.inverse.mel_to_stft(mel_np, sr=16000)
#y = librosa.griffinlim(sft)
#sf.write("a.wav", y, samplerate=16000)


# Torch mel
config_f = "/home/rosen/project/FastSpeech2/config/ESD/preprocess_iiv.yaml"
model_config_f = "/home/rosen/project/FastSpeech2/config/ESD/model_fastspeechIIV.yaml"
model_config = yaml.load(open(model_config_f, "r"), Loader=yaml.FullLoader)
preprocess_config = yaml.load(open(config_f, "r"), Loader=yaml.FullLoader)
stft = Audio.stft.TacotronSTFT(
    preprocess_config["preprocessing"]["stft"]["filter_length"],
    preprocess_config["preprocessing"]["stft"]["hop_length"],
    preprocess_config["preprocessing"]["stft"]["win_length"],
    preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
    preprocess_config["preprocessing"]["audio"]["sampling_rate"],
    preprocess_config["preprocessing"]["mel"]["mel_fmin"],
    preprocess_config["preprocessing"]["mel"]["mel_fmax"],
)

if False:
    #mel, energy = get_mel_from_wav(y, stft)

    mel = torch.from_numpy(np.load(mel_f2, allow_pickle=True)).T.float()
    #mel = torch.from_numpy(mel).float()

    #librosa.output.write_wav("a_conv1ds.wav", mel, 16000)
    #inv_mel_spec(mel, "b_conv1ds_from_mel.wav", stft)

if False:
    audio_path = "b_conv1ds_hifi_from_mel.wav"
    mel = torch.from_numpy(np.load(mel_f2, allow_pickle=True)).T.float()
    vocoder = get_vocoder(model_config, device)
    wav_reconstruction = vocoder_infer(
        mel.unsqueeze(0),
        vocoder,
        model_config,
        preprocess_config,
    )[0]
    write(audio_path, stft.sampling_rate, wav_reconstruction)

if True:
    audio_path = "b_conv1ds_hifi16k_from_mel.wav"
    from speechbrain.pretrained import HIFIGAN
    mel = torch.from_numpy(np.load(mel_f2, allow_pickle=True)).T.float()

    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir")
    #mel_specs = torch.rand(2, 80, 298)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel.unsqueeze(0))
    waveforms = waveforms.cpu().numpy()
    write(audio_path, stft.sampling_rate, waveforms)