import librosa
import librosa.feature as lf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def extract_mfccs(segment, sr, n_mfcc=13, n_fft=2048, hop_length=None):
    mfccs = lf.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs.T

def pad_mfccs(mfccs, max_length=156):
    if mfccs.shape[0] < max_length:
        padding = np.zeros((max_length - mfccs.shape[0], mfccs.shape[1]))
        mfccs = np.vstack((mfccs, padding))
    elif mfccs.shape[0] > max_length:
        mfccs = mfccs[:max_length, :]
    return mfccs

def generate_timecode(segment_index, total_segments, audio_duration, frame_rate=30):
    # Calculate the time in seconds for each segment
    time_per_segment = audio_duration / total_segments
    total_seconds = time_per_segment * segment_index

    # Extract hours, minutes, seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Calculate frames and sub-frames (milliseconds)
    fractional_seconds = total_seconds % 1
    frames = int(fractional_seconds * frame_rate)
    sub_frames = int((fractional_seconds * frame_rate - frames) * 1000)

    # Format timecode as HH:MM:SS:FF.FFF
    timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}.{sub_frames:03d}"
    return timecode


# Load the audio file
audio_path = "sneezes/test_audio/test_1.wav"
y, sr = librosa.load(audio_path, sr=None)

# Define window and hop size for audio segmentation
window_size = 0.1  # seconds
hop_size = 0.05   # seconds
window_samples = int(window_size * sr)
hop_samples = int(hop_size * sr)

# Segment the audio and extract features
segments = []
for start in range(0, len(y) - window_samples + 1, hop_samples):
    segment = y[start:start + window_samples]
    mfccs = extract_mfccs(segment, sr)
    segments.append(mfccs)

# Load your trained model
model = load_model("sneezes/models/model.h5")

# Predict facial parameters for each segment and aggregate predictions
predicted_face_data = []
for mfccs in segments:
    mfccs_padded = pad_mfccs(mfccs)
    mfccs_reshaped = np.reshape(mfccs_padded, (1, mfccs_padded.shape[0], mfccs_padded.shape[1]))
    prediction = model.predict(mfccs_reshaped)[0]
    aggregated_prediction = np.mean(prediction, axis=0)
    predicted_face_data.append(aggregated_prediction)

# Create DataFrame for predictions
columns = pd.read_csv('sneezes/face_parameters/number30_1.csv').columns.drop(['Timecode', 'BlendshapeCount'])
final_predictions_df = pd.DataFrame(predicted_face_data, columns=columns)

# Add timecodes and BlendshapeCount
audio_duration = librosa.get_duration(y=y, sr=sr)
final_predictions_df['Timecode'] = [generate_timecode(i, len(final_predictions_df), audio_duration) for i in range(len(final_predictions_df))]
final_predictions_df['BlendshapeCount'] = 61

# Reorder columns to put Timecode and BlendshapeCount first
final_predictions_df = final_predictions_df[['Timecode', 'BlendshapeCount'] + [col for col in final_predictions_df.columns if col not in ['Timecode', 'BlendshapeCount']]]

# Save the DataFrame to a CSV file
final_predictions_df.to_csv('C:/Users/xubor/OneDrive/Desktop/predicted_face_data.csv', index=False)

# print(final_predictions_df)