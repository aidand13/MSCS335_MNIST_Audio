import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import random

# your existing imports:
from MNIST_Audio_CNN import AudioCNN, AudioMNISTDataset, AudioTransform

# hyperparameter candidates
TIME_MASK_PARAMS = [5, 10, 15]    # up to ~5–15 frames masked (≈0.06–0.19 s)
FREQ_MASK_PARAMS = [4, 8, 12]     # up to ~4–12 mel‐bins masked

TRAIN_AUDIO_DIR = "../Organize_MNIST_Audio/MNIST_Audio_Train"
TEST_AUDIO_DIR = "../Organize_MNIST_Audio/MNIST_Audio_Test"

NUM_TO_SHOW = 5
SR = 8000
N_MELS = 64
TARGET_LENGTH = 6700
HOP_LENGTH = 100
N_FFT = 400


def train_and_eval(tm, fm, epochs=3, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build your transforms
    mel_tfm = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                               hop_length=HOP_LENGTH, n_mels=N_MELS,
                               center=False)
    db_tfm  = T.AmplitudeToDB()
    transform = AudioTransform(SR, TARGET_LENGTH, mel_tfm, db_tfm,
                               time_mask_param=tm,
                               freq_mask_param=fm)
    transform.training = True

    # datasets & loaders
    train_ds = AudioMNISTDataset(TRAIN_AUDIO_DIR, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True,
                              persistent_workers=True)

    # for validation, turn off augmentation
    transform.training = False
    val_ds = AudioMNISTDataset(TEST_AUDIO_DIR, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True)

    # model & optimizer
    model = AudioCNN(num_classes=10).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    # quick train loop
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    # evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total   += y.size(0)

    return correct/total

if __name__ == "__main__":
    # hyperparameter grid
    TIME_MASK_PARAMS = [5, 10, 15]
    FREQ_MASK_PARAMS = [4, 8, 12]

    results = {}
    for tm, fm in itertools.product(TIME_MASK_PARAMS, FREQ_MASK_PARAMS):
        acc = train_and_eval(tm, fm, epochs=3)
        results[(tm, fm)] = acc
        print(f"tm={tm}, fm={fm} → val_acc={acc:.4f}")

    best = max(results, key=results.get)
    print(f"Best params: time={best[0]}, freq={best[1]} → acc={results[best]:.4f}")