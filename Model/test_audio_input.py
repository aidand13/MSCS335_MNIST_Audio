import torchaudio
import matplotlib.pyplot as plt
import torchaudio.transforms as T

# Load the .wav file
waveform, sample_rate = torchaudio.load('output.wav')

# Apply a MelSpectrogram transform (optional, but often used for audio visualization)
transform = T.MelSpectrogram(sample_rate=sample_rate)
mel_spec = transform(waveform)

# Convert MelSpectrogram to decibel scale (dB)
mel_db = T.AmplitudeToDB()(mel_spec)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(mel_db[0].numpy(), cmap='inferno', aspect='auto', origin='lower',
           extent=(0.0, float(mel_db.size(1)), 0.0, float(mel_db.size(0))))
plt.title('Spectrogram')
plt.ylabel('Frequency Bin')
plt.xlabel('Time Frame')
plt.colorbar(format="%+2.0f dB")
plt.show()
