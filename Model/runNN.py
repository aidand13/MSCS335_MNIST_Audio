from MNIST_Audio_CNN import AudioCNN, AudioMNISTDataset, AudioTransform
from input_to_wav import get_user_input
from visualize import plot_spec, process_audio
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio.transforms as T
import torchaudio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

TRAIN_AUDIO_DIR = "../Organize_MNIST_Audio/MNIST_Audio_Train"
TEST_AUDIO_DIR = "../Organize_MNIST_Audio/MNIST_Audio_Test"
SR = 8000
N_MELS = 64
TARGET_LENGTH = 6700
HOP_LENGTH = 100
N_FFT = 400

MEL_TRANSFORM = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=False
    )

DB_TRANSFORM = T.AmplitudeToDB()

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

    transform = AudioTransform(
        sr=SR,
        target_length=TARGET_LENGTH,
        mel_transform=MEL_TRANSFORM,
        db_transform=DB_TRANSFORM
    )

    transform.training = True  # augmentation ON for train set

    # Load dataset with normalization
    train_dataset = AudioMNISTDataset(TRAIN_AUDIO_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() or 4,
                              pin_memory=True, persistent_workers=True)
    # num_workers: splits up data processing between cpu cores
    # pin_memory=True: use page‐locked (pinned) host memory to speed up .to(device) transfers
    # drop_last=True: if data size % batch size != 0, drop the last batch to keep all batches the same size. Never happens with batch size 8,16,32
    # persistent_workers=True: Epoch 1 takes awhile to start because the workers have to be created, this allows workers to transfer between epochs greatly decreasing runtime.

    test_loader = None

    if display_test_acc:
        transform.training = False
        test_dataset = AudioMNISTDataset(TEST_AUDIO_DIR, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() or 4,
                                 pin_memory=True, persistent_workers=True)

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

def run_trained_network(trained_network="MNIST_Audio_CNN.pt", batch_size=32, save_matrix=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = AudioTransform(SR, TARGET_LENGTH, MEL_TRANSFORM, DB_TRANSFORM)
    transform.training = False

    # Load test set
    test_dataset = AudioMNISTDataset(TEST_AUDIO_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = AudioCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(trained_network, map_location=device))
    model.eval()

    preds, labels = accuracy(model, test_loader, device)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(0,10))
    disp.plot()
    plt.title("Confusion Matrix")
    if save_matrix:
        plt.savefig('MNIST_Audio_Matrix.png')
    plt.show()

def classify_wav_file(file_path, trained_network="MNIST_Audio_CNN.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = AudioTransform(SR, TARGET_LENGTH, MEL_TRANSFORM, DB_TRANSFORM)

    # Load the model
    model = AudioCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(trained_network, map_location=device))
    model.eval()

    # Load and preprocess audio
    waveform, sr = torchaudio.load(file_path)
    mel_db_ = transform(waveform, sr)
    mel_db_ = mel_db_.unsqueeze(0).to(device)  # Add batch dim and move to device

    # Classify
    with torch.no_grad():
        output = model(mel_db_)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class

if __name__ == "__main__":
    print("1. Train CNN\t2. Test CNN\n3. Get and Classify User Input\t4. Test Output File")
    num = int(input("Select an option:\n"))
    if num==1:
        trainNN(batch_size=32, display_test_acc=True)
    elif num==2:
        run_trained_network(batch_size=32, save_matrix=True)
    elif num==3:
        get_user_input()

    if num==3 or num==4:
        pred = classify_wav_file("output.wav")
        print("Enter the number you spoke:")
        num = input()
        print(f"Actual: {num}\tPredicted: {pred}")
        db_transform=T.AmplitudeToDB()
        mel_db=db_transform(process_audio("output.wav"))
        plot_spec(mel_db)
