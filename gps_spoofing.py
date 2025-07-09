import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

df = pd.read_csv("airsim_rec.txt", sep=r'\s+')
positions = df[['POS_X', 'POS_Y', 'POS_Z']].values


# 1. Normalize the Data
scaler = StandardScaler()
positions_scaled = scaler.fit_transform(positions)


# 2. Create Sequences
def create_sequences(data, window_size=10):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)


X = create_sequences(positions_scaled, window_size=10)


# 3. LSTM Autoencoder
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    RepeatVector(X.shape[1]),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])

model.compile(optimizer='adam', loss='mse')
model.summary()


# 4. Train the Model
model.fit(X, X, epochs=50, batch_size=32, validation_split=0.1)


# 5. Reconstruction Error to Detect Anomalies
X_pred = model.predict(X)
reconstruction_error = np.mean(np.square(X - X_pred), axis=(1,2))

threshold = np.mean(reconstruction_error) + 3*np.std(reconstruction_error)
anomalies = reconstruction_error > threshold


# Print Anomalies
for i, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        print(f"Anomaly at sequence index {i}, position: {positions[i+10]}")


plt.figure(figsize=(12, 4))
plt.plot(reconstruction_error, label="Reconstruction Error")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.title("Reconstruction Error per Sequence")
plt.xlabel("Sequence Index")
plt.ylabel("MSE")
plt.legend()
plt.show()