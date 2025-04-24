import sounddevice as sd
from scipy.io.wavfile import write

sr = 8000  # Sample rate
seconds = 3  # Duration of recording

print("Recording...")
myrecording = sd.rec(int(seconds * sr), samplerate=sr, channels=2)
sd.wait()  # Wait until recording is finished
print("Recording complete")

write('output.wav', sr, myrecording)  # Save as WAV file
print("File saved as output.wav")