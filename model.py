import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from time_synchronize import synchronize
from utility import generate_timecode, add_timecode_column

face_directory = Path("sneezes/face_parameters")
audio_directory = Path("sneezes/sneeze_audio")
audio_paths = [str(p).replace('\\', '/') for p in audio_directory.rglob('*') if p.is_file()]
face_paths = [str(p).replace('\\', '/') for p in face_directory.rglob('*') if p.is_file()]
audio_data, face_data = [], []
columns = pd.read_csv(face_paths[0]).columns.drop(['BlendshapeCount'])

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

if len([m for m in models_path.glob('*')]) > 1:
    model = load_model(model_path)
else:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        TimeDistributed(Dense(y_train.shape[2]))  # Apply Dense layer to each time step
    ])

    # Model architecture with dropout and batch normalization
    # model = Sequential([
    #     LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
    #          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    #     Dropout(0.5),
    #     BatchNormalization(),
    #     LSTM(100, return_sequences=True),
    #     Dropout(0.5),
    #     BatchNormalization(),
    #     TimeDistributed(Dense(y_train.shape[2]))
    # ])
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    model.save(model_path)

test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

sample_index = 0
sample_audio = X_test[sample_index:sample_index + 1]
print(sample_audio)
predicted_face_data = model.predict(sample_audio)[0]

print(sample_audio.shape)
print(predicted_face_data.shape)

print("Predicted Facial Parameters for the Sample:")
processed_prediction = pd.DataFrame(predicted_face_data, columns=columns[1:])
new_prediction = add_timecode_column(processed_prediction)
print(new_prediction)

# Save the modified DataFrame to a new CSV file
output_path = "C:/Users/xubor/OneDrive/Desktop/output.csv"
new_prediction.to_csv(output_path, index=False)
print(f"Timecode added and saved to {output_path}")
