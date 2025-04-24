from MNIST_Audio_CNN import AudioCNN, AudioMNISTDataset, AudioTransform
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio.transforms as T
import torchaudio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

TRAIN_AUDIO_DIR = "Organize_MNIST_Audio/MNIST_Audio_Train"
TEST_AUDIO_DIR = "Organize_MNIST_Audio/MNIST_Audio_Test"
SR = 8000
N_MELS = 64
TARGET_LENGTH = 6700
HOP_LENGTH = 100
N_FFT = 400

def accuracy(model, test_loader_, device):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader_:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Accuracy on test set: {correct / total:.4f}")

    return all_preds, all_labels

def trainNN(epochs=15, batch_size=16, lr=0.001, display_test_acc=False, save_file="MNIST_Audio_CNN.pt"):

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

def run_trained_network(trained_network="MNIST_Audio_CNN.pt", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    mel_transform = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=False
    )
    db_transform = T.AmplitudeToDB()
    transform = AudioTransform(SR, TARGET_LENGTH, mel_transform, db_transform)

    # Load test set
    test_dataset = AudioMNISTDataset(TEST_AUDIO_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = AudioCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(trained_network, map_location=device))
    model.eval()

    preds, labels = accuracy(model, test_loader, device)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(1,11))
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

def classify_wav_file(file_path, trained_network="MNIST_Audio_CNN.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms (same as training)
    mel_transform = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                     n_mels=N_MELS, center=False)
    db_transform = T.AmplitudeToDB()
    transform = AudioTransform(SR, TARGET_LENGTH, mel_transform, db_transform)

    # Load the model
    model = AudioCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(trained_network, map_location=device))
    model.eval()

    # Load and preprocess audio
    waveform, sr = torchaudio.load(file_path)
    mel_db = transform(waveform, sr)
    mel_db = mel_db.unsqueeze(0).to(device)  # Add batch dim and move to device

    # Classify
    with torch.no_grad():
        output = model(mel_db)
        predicted_class = torch.argmax(output, dim=1).item()

    # print(f"Predicted class: {predicted_class}")
    return predicted_class

if __name__ == "__main__":
    # trainNN(batch_size=32, display_test_acc=False)
    # run_trained_network(batch_size=32)

    num = np.random.randint(0, 10)
    person = np.random.randint(55, 61)
    take = np.random.randint(0, 50)

    filename = f"../Organize_MNIST_Audio/MNIST_Audio_Test/{num}_{person}_{take}.wav"

    print(f"Classifying: {num}_{person}_{take}.wav")
    pred = classify_wav_file(filename)
    print(f"{num}:{pred}")