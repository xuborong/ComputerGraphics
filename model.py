from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from time_synchronize import synchronize

X_train, X_val, X_test = ...
y_train, y_val, y_test = ...

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(y_train.shape[1]))  # Output layer with a unit for each facial parameter

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
