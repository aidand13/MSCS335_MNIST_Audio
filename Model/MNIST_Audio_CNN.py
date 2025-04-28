import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import os
import torchaudio
import torchaudio.transforms as T
import random

class AudioTransform:
    def __init__(self, sr, target_length, mel_transform, db_transform, time_mask_param=9,freq_mask_param=10):
        self.sr = sr
        self.target_length = target_length
        self.mel_transform = mel_transform
        self.db_transform = db_transform

        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

        self.training = False
        #self.mask_count = 0

    def __call__(self, waveform, orig_sr):
        # Resample if needed
        if orig_sr != self.sr:
            resample = T.Resample(orig_freq=orig_sr, new_freq=self.sr)
            waveform = resample(waveform)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize to [-1, 1]
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        # Pad or trim
        if waveform.size(1) < self.target_length:
            waveform = F.pad(waveform, (0, self.target_length - waveform.size(1)))
        else:
            waveform = waveform[:, :self.target_length]

        # Mel spectrogram + dB
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        if self.training and random.random() < 0.5:
            #self.mask_count += 1

            mel_db = self.time_mask(mel_db)
            mel_db = self.freq_mask(mel_db)

        return mel_db

class AudioMNISTDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.files = [f for f in os.listdir(audio_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.audio_dir, self.files[idx])
        waveform, sr = torchaudio.load(path)

        # Apply preprocessing
        mel_db = self.transform(waveform, sr)

        # Extract label from filename
        label = int(self.files[idx][0])

        return mel_db, label

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()

        # Convolutional Layers
        self.in_to_h1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))  # -> [B, 16, 64, 64]
        self.norm = nn.BatchNorm2d(16)

        self.h1_to_h2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))  # -> [B, 32, 32, 32]

        # Fully Connected Layers
        self.h2_to_h3 = nn.Linear(32 * 16 * 16, 128)
        self.h3_to_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.in_to_h1(x)
        x = F.relu(self.norm(x))  # [B, 16, 64, 64]
        x = F.max_pool2d(x, (2, 2))     # [B, 16, 32, 32]
        x = F.relu(self.h1_to_h2(x))  # [B, 32, 32, 32]
        x = F.max_pool2d(x, (2, 2))     # [B, 32, 16, 16]

        x = torch.flatten(x, 1)     # [B, 32*16*16]
        x = F.relu(self.h2_to_h3(x))        # [B, 128]
        x = F.dropout(x, 0.1)  # drop out 10% of the channels
        x = self.h3_to_out(x)       # [B, num_classes]

        return x
