"""
The purpose of this file is work with and  test different padding and sizes
for our spectrograms. It uses the same normalization in the waveform that
we will use for our neural network.
"""

import os
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio as ta
import matplotlib.pyplot as plt

AUDIO_DIR = "../Organize_MNIST_Audio/MNIST_Audio_Train"  # or MNIST_Audio_Test

# Mel parameters
SR = 8000
N_MELS = 64
HOP_LENGTH = 100
N_FFT = 400

# Padding length
TARGET_LENGTH = 6700

# Testing amount
NUM_TO_SHOW = 5

n_frames = (TARGET_LENGTH - N_FFT) // HOP_LENGTH + 1
time_mask_param = int(n_frames * 0.15)  # ~9 frames (~0.12 s)
freq_mask_param = int(N_MELS * 0.15)   # ~10 bins



mel_transform = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    center=False
)
db_transform = T.AmplitudeToDB()

def process_audio(path_):
    waveform, sr = ta.load(path_)

    if sr != SR:
        resample = T.Resample(orig_freq=sr, new_freq=SR)
        waveform = resample(waveform)

    if waveform.size(dim=0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform / waveform.abs().max()

    length = waveform.size(1)
    if length < TARGET_LENGTH:
        padding = TARGET_LENGTH - length
        waveform = F.pad(waveform, (0, padding))
    elif length > TARGET_LENGTH:
        waveform = waveform[:, :TARGET_LENGTH]

    return mel_transform(waveform)

def plot_spec(mel_db_):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db_[0], aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='db')
    plt.title('Mel-Spectrogram (dB)')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Grab first few .wav files
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")][:NUM_TO_SHOW]

    for fname in wav_files:
        path = os.path.join(AUDIO_DIR, fname)
        mel_spec = process_audio(path)
        print(mel_spec.shape)
        mel_db = db_transform(mel_spec)

        plot_spec(mel_db)

