import yfinance as yf
data = yf.download('BTC-USD', start='2020-01-01', end='2023-12-31', interval='1d')
# Moving Average (MA)
data['MA_7'] = data['Close'].rolling(window=7).mean()
data['MA_30'] = data['Close'].rolling(window=30).mean()

# RSI (Relative Strength Index)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
data['RSI'] = 100 - (100 / (1 + gain / loss))

# MACD
data['EMA_12'] = data['Close'].ewm(span=12).mean()
data['EMA_26'] = data['Close'].ewm(span=26).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']])

# Tạo dataset cho LSTM (dùng sliding window)
def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(len(data)-window_size-1):
        X.append(data[i:(i+window_size), 0])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Xây dựng mô hình LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)
model.save('crypto_predictor.h5')
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('crypto_predictor.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Nhận dữ liệu giá gần nhất
    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
    prediction = model.predict(scaled_data.reshape(1, 60, 1))
    return jsonify({"prediction": scaler.inverse_transform(prediction)[0][0]})

if __name__ == '__main__':
    app.run()

<!---
pptrinhan/pptrinhan is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
