import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa

OUTPUT_FILENAME = 'output.wav'
SAMPLE_RATE = 8000  # 8kHz sampling rate
DURATION = 3    # Recording length


def record_audio():
    print(f"Recording for {DURATION} seconds...")
    audio_ = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    print("Recording done!")
    return audio_.flatten()


def trim_to_voiced(audio_, sr, target_length=6700):
    audio_ = audio_.astype(np.float32) / 32768.0
    # Remove silence
    trimmed, _ = librosa.effects.trim(audio_, top_db=30)

    if len(trimmed) > target_length:
        trimmed = trimmed[:target_length]
    else:
        trimmed = np.pad(trimmed, (0, target_length - len(trimmed)), mode='constant')

    trimmed = (trimmed * 32767).astype(np.int16)
    return trimmed

def get_user_input():
    print(f"Please speak a number 0-9")
    audio = record_audio()
    trimmed_audio = trim_to_voiced(audio, SAMPLE_RATE)
    wav.write(OUTPUT_FILENAME, SAMPLE_RATE, trimmed_audio)

if __name__ == "__main__":
    get_user_input()


