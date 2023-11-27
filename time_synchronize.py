import pandas as pd
import librosa
import librosa.feature as lf
import numpy as np


def synchronize(audio_file, face_file):
    y, sr = librosa.load(audio_file, sr=None)
    facial_data = pd.read_csv(face_file).drop("BlendshapeCount", axis=1)
    audio_data = pd.DataFrame([])

    def timecode_to_seconds(timecode):
        hours, minutes, seconds, milli_fraction = timecode.split(':')
        milliseconds, fraction = milli_fraction.split('.')
        total_seconds = (int(hours) * 3600 +
                         int(minutes) * 60 +
                         int(seconds) +
                         int(milliseconds) / 1000 +
                         int(fraction) / 1000000)
        return total_seconds

    def extract_mfcc_for_frame(frame_number, window_size=0.1):
        start_time = max(frame_number - window_size / 2, 0)
        end_time = start_time + window_size
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        segment = y[start_sample:end_sample]
        segment_mfccs = lf.mfcc(y=segment, sr=sr, n_mfcc=13)
        avg_mfccs = np.mean(segment_mfccs, axis=1)
        return avg_mfccs

    time_in_seconds = facial_data['Timecode'].apply(timecode_to_seconds)
    start_time_seconds = time_in_seconds.iloc[0]
    time_offset = time_in_seconds - start_time_seconds
    audio_frames = time_offset * sr
    audio_data["MFCC"] = audio_frames.apply(lambda frame: extract_mfcc_for_frame(frame / sr))
    facial_data = facial_data.drop("Timecode", axis=1)

    return facial_data, audio_data


audio_path = "sneezes/sneeze_audio/sneeze_1.wav"
face_path = "sneezes/face_parameters/sneeze_1.csv"
face, audio = synchronize(audio_path, face_path)
