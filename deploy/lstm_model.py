import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import SVG
from keras.utils import model_to_dot
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def app():
    st.title("Análisis de series de tiempo con LSTM")
    st.write("Este es un ejemplo de análisis de series de tiempo utilizando LSTM (Long Short-Term Memory) en Streamlit.")

    # Set start and end dates for the price data
    start_date = st.date_input('Start Train' , value=pd.to_datetime('2018-1-1'))
    end_date = st.date_input('End Train' , value=pd.to_datetime('today'))


    # Retrieve the Bitcoin price data from Yahoo Finance
    i01_BTC_USD = yf.download('BTC-USD', start=start_date, end=end_date)
    i01_BTC_USD.drop('Volume', axis=1, inplace=True)
    i01_BTC_USD.columns = ['BTC_' + column for column in i01_BTC_USD.columns]

    # Retrieve the Ethereum price data from Yahoo Finance
    i02_ETH_USD = yf.download('ETH-USD', start=start_date, end=end_date)
    i02_ETH_USD.drop('Volume', axis=1, inplace=True)
    i02_ETH_USD.columns = ['ETH_' + column for column in i02_ETH_USD.columns]

    # Concatenate the DataFrames
    df = pd.concat([i02_ETH_USD, i01_BTC_USD], axis=1)

    # Preprocessing
    df['BTC_Return'] = df['BTC_Close'].pct_change()
    df['Trend'] = np.where(df['BTC_Return'] > 0.00, 1, 0)

    # Drop rows with missing values
    df = df.dropna()

    # Train-test split
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data.drop(['BTC_Return'], axis=1))
    scaled_test_data = scaler.transform(test_data.drop(['BTC_Return'], axis=1))

    # Create time sequences
    window_size = 100

    def create_sequences(data):
        x = []
        y = []
        for i in range(window_size, len(data)):
            x.append(data[i - window_size:i])
            y.append(data[i][-1])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(scaled_train_data)
    x_test, y_test = create_sequences(scaled_test_data)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=160, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=160))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test), verbose=1)

    # Display model summary
    st.subheader("Resumen del modelo LSTM")
    st.text(str(model.summary()))

    # Display model architecture diagram
    st.subheader("Arquitectura del modelo LSTM")
    graph = model_to_dot(model, show_shapes=True, show_dtype=False, show_layer_names=True, rankdir="TB",
                         expand_nested=False, dpi=70, subgraph=False).create(prog='dot', format='svg')
    st.image(graph)