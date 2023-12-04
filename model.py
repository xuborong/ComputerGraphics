import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from sklearn.model_selection import train_test_split
from time_synchronize import synchronize
from utility import generate_timecode, add_timecode_column

face_directory = Path("sneezes/face_parameters")
audio_directory = Path("sneezes/sneeze_audio")
audio_paths = [str(p).replace('\\', '/') for p in audio_directory.rglob('*') if p.is_file()]
face_paths = [str(p).replace('\\', '/') for p in face_directory.rglob('*') if p.is_file()]
audio_data, face_data = [], []
columns = pd.read_csv(face_paths[0]).columns

for audio_path, face_path in zip(audio_paths, face_paths):
    face, audio = synchronize(audio_path, face_path)
    audio_data.append(audio)
    face_data.append(face)

max_audio_length = max(array.shape[0] for array in audio_data)
max_face_length = max(array.shape[0] for array in face_data)

audio_data_padded = np.array([np.pad(item, ((0, max_audio_length - item.shape[0]), (0, 0)), 'constant') for item in audio_data])
face_data_padded = np.array([np.pad(item, ((0, max_face_length - item.shape[0]), (0, 0)), 'constant') for item in face_data])

X_temp, X_test, y_temp, y_test = train_test_split(audio_data_padded, face_data_padded, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)

models_path = Path("sneezes/models")
model_path = "sneezes/models/model.h5"

if len([m for m in models_path.glob('*')]) != 0:
    model = load_model(model_path)
else:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        TimeDistributed(Dense(y_train.shape[2]))  # Apply Dense layer to each time step
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    model.save(model_path)

test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

sample_index = 0
sample_audio = X_test[sample_index:sample_index + 1]
predicted_face_data = model.predict(sample_audio)[0]

print("Predicted Facial Parameters for the Sample:")
processed_prediction = pd.DataFrame(predicted_face_data, columns=columns[1:])
new_prediction = add_timecode_column(processed_prediction)
print(new_prediction)
