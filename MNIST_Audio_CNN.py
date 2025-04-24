import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torchaudio
import torchaudio.transforms as T

TRAIN_AUDIO_DIR = "Organize_MNIST_Audio/MNIST_Audio_Train"
TEST_AUDIO_DIR = "Organize_MNIST_Audio/MNIST_Audio_Test"
SR = 8000
N_MELS = 64
TARGET_LENGTH = 6700
HOP_LENGTH = 100
N_FFT = 400

class AudioTransform:
    def __init__(self, sr, target_length, mel_transform, db_transform):
        self.sr = sr
        self.target_length = target_length
        self.mel_transform = mel_transform
        self.db_transform = db_transform

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


def accuracy(model, test_loader_, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader_:
            x, y = x.to(device), y.to(device)
            predictions = torch.argmax(model(x), dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    print(f"Accuracy on test set: {correct / total:.4f}")

def trainNN(epochs=15, batch_size=32, lr=0.001, display_test_acc=False, save_file="MNIST_Audio_CNN.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=False
    )

    db_transform = T.AmplitudeToDB()

    transform = AudioTransform(
        sr=SR,
        target_length=TARGET_LENGTH,
        mel_transform=mel_transform,
        db_transform=db_transform
    )

    # Load dataset with normalization
    train_dataset = AudioMNISTDataset(TRAIN_AUDIO_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = None

    if display_test_acc:
        test_dataset = AudioMNISTDataset(TEST_AUDIO_DIR, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, etc.
    model = AudioCNN(num_classes=10).to(device)
    print(f"Total parameters: {sum(param.numel() for param in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cross_entropy = nn.CrossEntropyLoss()

    running_loss = 0.0
    # Train
    for epoch in range(epochs):
        for _, data in enumerate(tqdm(train_loader)):
            x, y = data

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            accuracy(model, test_loader, device)

    torch.save(model.state_dict(), save_file)


if __name__ == "__main__":
    trainNN(display_test_acc=False)
