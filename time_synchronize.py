import pandas as pd
import librosa
import librosa.feature as lf
import numpy as np

def synchronize(audio_file, face_file):
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)

    # Load and preprocess facial data
    facial_data = pd.read_csv(face_file)
    facial_data['Time_in_Seconds'] = facial_data['Timecode'].apply(timecode_to_seconds)
    start_time_seconds = facial_data['Time_in_Seconds'].iloc[0]
    facial_data['Time_Offset'] = facial_data['Time_in_Seconds'] - start_time_seconds

    # Drop non-feature columns
    facial_data = facial_data.drop(["Timecode", "Time_in_Seconds", "BlendshapeCount"], axis=1)

    # Initialize audio feature array
    mfccs = []

    # Extract MFCC for each frame in facial data
    for offset in facial_data['Time_Offset']:
        frame_number = int(offset * sr)
        mfcc = extract_mfcc_for_frame(y, sr, frame_number)
        mfccs.append(mfcc)

    # Convert list of MFCCs to a consistent 2D numpy array
    mfccs_array = np.vstack(mfccs)

    return facial_data, mfccs_array

def timecode_to_seconds(timecode):
    hours, minutes, seconds, milli_fraction = timecode.split(':')
    milliseconds, fraction = milli_fraction.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000 + int(fraction) / 1000000

def extract_mfcc_for_frame(audio, sr, frame_number, window_size=0.1, n_mfcc=13, default_n_fft=2048):
    start_time = max(frame_number - window_size / 2, 0)
    end_time = start_time + window_size
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    segment = audio[start_sample:end_sample]

    # Handle zero-length segment
    if len(segment) == 0:
        return np.zeros(n_mfcc)

    # Dynamically adjust n_fft and extract MFCCs
    n_fft = min(len(segment), default_n_fft)
    segment_mfccs = lf.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    avg_mfccs = np.mean(segment_mfccs, axis=1)
    return avg_mfccs

audio_path = "sneezes/sneeze_audio/sneeze_1.wav"
face_path = "sneezes/face_parameters/sneeze_1.csv"
face, audio = synchronize(audio_path, face_path)
print("Facial Data Shape:", face.shape)
print("Audio Data Shape:", audio.shape)
