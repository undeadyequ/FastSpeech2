import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from audio.stft import STFT, TacotronSTFT
from audio.tools import get_mel_from_wav
import audio as Audio

wav_f1 = "/home/rosen/project/FastSpeech2/0014_001606.wav"

audio_org = librosa.load(wav_f1, sr=None)[0]
device = 'cpu'
filter_length = 1024
hop_length = 256
win_length = 1024 # doesn't need to be specified. if not specified, it's the same as filter_length
window = 'hann'
sr = 16000
#librosa_stft = librosa.stft(audio_org, n_fft=filter_length, hop_length=hop_length)
mel_libro = librosa.feature.melspectrogram(audio_org, sr=sr, n_fft=1024, n_mels=80, hop_length=256)

#_magnitude = np.abs(librosa_stft)

audio = torch.FloatTensor(audio_org)
audio = audio.unsqueeze(0)
audio = audio.to(device)

stft = STFT(
    filter_length=filter_length,
    hop_length=hop_length,
    win_length=win_length,
    window=window
).to(device)

magnitude, phase = stft.transform(audio)

stftTactron = Audio.stft.TacotronSTFT(
    1024,
    256,
    1024,
    80,
    16000,
    0,
    8000)

mel_con1d, energy = get_mel_from_wav(audio_org, stftTactron)

#mel = torch.from_numpy(np.load(mel_f1, allow_pickle=True)).T.float()
mel_con1d = torch.from_numpy(mel_con1d).float()

def show_mel(mel_1, mel_2):
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.title("PyTorch STFT magnitude")
    plt.xlabel('Frames')
    plt.ylabel('FFT bin')

    #plt.imshow(20*np.log10(1+magnitude[0].cpu().data.numpy()), aspect='auto', origin='lower')
    plt.imshow(mel_1, aspect='auto', origin='lower')

    plt.subplot(212)
    plt.title('Librosa STFT magnitude')
    plt.xlabel('Frames')
    plt.ylabel('FFT bin')
    #plt.imshow(20*np.log10(1+_magnitude), aspect='auto', origin='lower')
    plt.imshow(mel_2 * 32768.0, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.savefig('stft.png')


if __name__ == '__main__':
    show_mel(mel_con1d, mel_libro)