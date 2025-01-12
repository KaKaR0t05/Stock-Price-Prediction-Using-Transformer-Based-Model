import yfinance as yf
import ta
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Add, GlobalAveragePooling1D
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from keras.utils import plot_model

import matplotlib.pyplot as plt

# Constants
TRAIN_DATA_RATIO = 0.8
VALIDATION_DATA_RATIO = 0.2
NUMBER_OF_SERIES_FOR_PREDICTION = 24

# Download S&P 500 Data
reliance_data = yf.download('RELIANCE.NS', interval='5m', period='1mo')
# file_path = r'reliance_data.csv'
# reliance_data = pd.read_csv(file_path)

# Feature extraction
feature = pd.DataFrame(index=reliance_data.index)
# Feature extraction
feature['SMA'] = ta.trend.sma_indicator(reliance_data['Close'].squeeze(), window=14)
feature['MACD'] = ta.trend.macd(reliance_data['Close'].squeeze())
feature['RSI'] = ta.momentum.rsi(reliance_data['Close'].squeeze())
feature['Close'] = reliance_data['Close']
# feature['SMA_50'] = ta.trend.sma_indicator(reliance_data['Close'].squeeze(), window=50)
# feature['EMA_200'] = ta.trend.ema_indicator(reliance_data['Close'].squeeze(), window=200)
# feature['Bollinger_Bands'] = ta.volatility.bollinger_hband(reliance_data['Close'].squeeze(), window=20)

# Normalization
mean = feature.mean()
std = feature.std()
feature = (feature - mean) / std

# Drop NaN values
feature = feature.dropna()
reliance_data = reliance_data.loc[feature.index]

# Train-test split
train_data_size = int(len(feature) * TRAIN_DATA_RATIO)
train = feature[:train_data_size]
test = feature[train_data_size:]

# Dataset preparation
def create_dataset(dataset, series_length):
    X, y = [], []
    for i in range(len(dataset) - series_length):
        X.append(dataset.iloc[i:i + series_length].values)
        y.append(dataset.iloc[i + series_length, -1])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train, NUMBER_OF_SERIES_FOR_PREDICTION)
X_test, y_test = create_dataset(test, NUMBER_OF_SERIES_FOR_PREDICTION)

# Transformer block definition
def transformer_block(inputs, model_dim, num_heads, ff_dim, dropout=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = Dense(model_dim)(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

# Positional encoding
def positional_encoding(max_position, model_dim):
    angle_rads = np.arange(max_position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(model_dim) // 2)) / model_dim)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

# Model building
def build_transformer_model(input_shape, model_dim, num_heads, num_layers, ff_dim, output_dim, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = Dense(model_dim)(inputs)
    position_encoding = positional_encoding(input_shape[0], model_dim)
    x = x + position_encoding

    for _ in range(num_layers):
        x = transformer_block(x, model_dim, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim)(x)
    return Model(inputs, outputs)

input_shape = (X_train.shape[1], X_train.shape[2])
model = build_transformer_model(input_shape, 256, 8, 6, 512, 1)

# Model compilation
model.compile(optimizer=Adam(), loss=MeanAbsoluteError(), metrics=['mae','mse'])

# Learning rate scheduler
def custom_lr_schedule(epoch, lr):
    warmup_epochs = 10
    warmup_lr = 1e-4
    initial_lr = 1e-3
    decay_rate = 0.4
    decay_step = 10
    if epoch < warmup_epochs:
        return warmup_lr + (initial_lr - warmup_lr) * (epoch / warmup_epochs)
    else:
        return initial_lr * (decay_rate ** ((epoch - warmup_epochs) // decay_step))

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('model_checkpoint.keras', save_best_only=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    LearningRateScheduler(custom_lr_schedule)
]

# Training
history = model.fit(X_train, y_train, validation_split=VALIDATION_DATA_RATIO, epochs=100, batch_size=128, callbacks=callbacks)

# Evaluation
loss, mae, mse = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}, Test MSE : {mse}")
# Predictions
predictions = model.predict(X_test)
predictions = predictions.flatten() * std['Close'] + mean['Close']
actual = reliance_data['Close'].iloc[train_data_size + NUMBER_OF_SERIES_FOR_PREDICTION:].values

# Visualization
# Plot Training and Validation Loss
plt.figure(figsize=(15, 8))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Actual vs Predicted Price Plot
plt.figure(figsize=(15, 8))
plt.plot(actual, label='Actual Close Prices')
plt.plot(predictions, label='Predicted Close Prices', linestyle='--')
plt.title("Actual vs Predicted Close Prices")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()
#plot_model(model, to_file='model_architecture.png', show_shapes=True)
# Save model
model.save('my_model.keras')
