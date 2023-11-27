import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from time_synchronize import synchronize

# Define directories
face_directory = Path("sneezes/face_parameters")
audio_directory = Path("sneezes/sneeze_audio")

# Gather file paths
audio_paths = [str(p).replace('\\', '/') for p in audio_directory.rglob('*') if p.is_file()]
face_paths = [str(p).replace('\\', '/') for p in face_directory.rglob('*') if p.is_file()]

# Initialize data arrays
audio_data, face_data = [], []

# Synchronize audio and face data
for audio_path, face_path in zip(audio_paths, face_paths):
    face, audio = synchronize(audio_path, face_path)
    audio_data.append(audio)
    face_data.append(face)

# Find the maximum length of sequences in audio_data and face_data
max_audio_length = max(array.shape[0] for array in audio_data)
max_face_length = max(array.shape[0] for array in face_data)

# Pad audio_data and face_data to the maximum length
audio_data_padded = np.array([np.pad(item, ((0, max_audio_length - item.shape[0]), (0, 0)), 'constant') for item in audio_data])
face_data_padded = np.array([np.pad(item, ((0, max_face_length - item.shape[0]), (0, 0)), 'constant') for item in face_data])

# Reshape the target data (face_data) to match the output shape of the model
# Assuming each facial data sample should be a flat array
face_data_flattened = face_data_padded.reshape(face_data_padded.shape[0], -1)

# Split data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(audio_data_padded, face_data_flattened, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)

# Model architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(y_train.shape[1])  # Adjusted to match the flattened facial data
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
