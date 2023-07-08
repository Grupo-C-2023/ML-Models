import streamlit as st

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# To ignore warnings
import warnings

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

warnings.filterwarnings("ignore")
import yfinance as yf

def app():
    st.title('Modelo SVC')

    # Definir fechas de inicio y finalización para los datos de precios
    start_date = st.date_input('Start Train' , value=pd.to_datetime('2018-1-1'))
    end_date = st.date_input('End Train' , value=pd.to_datetime('today'))


    # Descargar datos de precios de Bitcoin utilizando Yahoo Finance
    df = yf.download('BTC-USD', start=start_date, end=end_date)

    st.subheader('Datos de precios de Bitcoin')
    st.write(df)

    # Crear variables predictoras
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    # Almacenar todas las variables predictoras en una variable X
    X = df[['Open-Close', 'High-Low']]
    st.subheader('Variables predictoras')
    st.write(X.head())

    # Variables objetivo
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    st.subheader('Variables objetivo')
    st.write(y)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    split_percentage = 0.8
    split = int(split_percentage * len(df))

    # Conjunto de datos de entrenamiento
    X_train = X[:split]
    y_train = y[:split]

    # Conjunto de datos de prueba
    X_test = X[split:]
    y_test = y[split:]

    # Crear y entrenar el clasificador de vectores de soporte
    cls = SVC().fit(X_train, y_train)

    # Predecir las señales utilizando el modelo entrenado
    df['Predicted_Signal'] = cls.predict(X)
    st.subheader('Señales predichas')
    st.write(df['Predicted_Signal'])

    # Calcular los rendimientos diarios
    df['Return'] = df.Close.pct_change()
    st.subheader('Rendimientos diarios')
    st.write(df['Return'])

    # Calcular los rendimientos de la estrategia
    df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
    st.subheader('Rendimientos de la estrategia')
    st.write(df['Strategy_Return'])

    # Calcular los rendimientos acumulados
    df['Cum_Ret'] = df['Return'].cumsum()
    st.subheader('Rendimientos acumulados')
    st.write(df)

    # Calcular los rendimientos acumulados de la estrategia
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    st.subheader('Rendimientos acumulados de la estrategia')
    st.write(df)

    # Graficar los rendimientos acumulados de la estrategia
    fig = plt.figure(figsize=(16, 8))
    plt.plot(df['Cum_Ret'], color='red', label='Rendimientos acumulados')
    plt.plot(df['Cum_Strategy'], color='blue', label='Rendimientos acumulados de la estrategia')
    plt.legend()
    plt.title('Rendimientos acumulados vs. Rendimientos acumulados de la estrategia')
    st.subheader('Gráfico de rendimientos acumulados')
    st.pyplot(fig)
