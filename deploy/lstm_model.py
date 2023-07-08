import matplotlib.pyplot as plt # Importación de la biblioteca matplotlib.pyplot para graficar
import pandas as pd # Importación de la biblioteca pandas para manipulación de datos en forma d
import numpy as np # Importación de la biblioteca numpy para operaciones numéricas

pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S') # Se capta la fecha y hora actual

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Instalación de la biblioteca yfinance
# !pip install yfinance
# yfinance es una biblioteca utilizada para obtener datos financieros de Yahoo Finance.
# Es necesario instalarla para poder utilizarla en el código.

import yfinance as yf

def app():

  # Set start and end dates for the price data
  start_date = '2018-01-01' # Se establece la fecha de inicio para los datos de precios
  end_date = '2022-12-31' # Se establece la fecha de fin para los datos de precios

  # Retrieve the Bitcoin price data from Yahoo Finance
  i01_BTC_USD = yf.download('BTC-USD', start=start_date, end=end_date)
  # Remove the 'Volume' column
  i01_BTC_USD.drop('Volume', axis=1, inplace=True)

  # Add 'BTC_' prefix to column names
  i01_BTC_USD.columns = ['BTC_' + column for column in i01_BTC_USD.columns]
  # Se obtienen los datos de precios de Bitcoin de Yahoo Finance
  i01_BTC_USD.sample(2)

  # Retrieve the Bitcoin price data from Yahoo Finance
  i02_ETH_USD = yf.download('ETH-USD', start=start_date, end=end_date)
  # Remove the 'Volume' column
  i02_ETH_USD.drop('Volume', axis=1, inplace=True)
  # Add 'BTC_' prefix to column names
  i02_ETH_USD.columns = ['ETH_' + column for column in i02_ETH_USD.columns]
  # Se obtienen los datos de precios de Bitcoin de Yahoo Finance
  i02_ETH_USD.sample(2)

  # Concatenate the DataFrames
  df = pd.concat([i02_ETH_USD, i01_BTC_USD], axis=1)

  df.sample(2)

  # Retrieve the Bitcoin price data from Yahoo Finance
  i03_GLD = yf.download('GLD', start=start_date, end=end_date)
  # Remove the 'Volume' column
  i03_GLD.drop('Volume', axis=1, inplace=True)
  # Add 'BTC_' prefix to column names
  i03_GLD.columns = ['GLD_' + column for column in i03_GLD.columns]
  # Se obtienen los datos de precios de Bitcoin de Yahoo Finance
  i03_GLD.sample(2)

  # Concatenate the DataFrames
  df02 = pd.concat([i03_GLD , df], axis=1)
  # Preprocesamiento - Paso 1:

  # Preprocesamiento de los datos
  df02['BTC_Return'] = df02['BTC_Close'].pct_change()

  # Definir la tendencia positiva
  df02['Trend'] = np.where(df02['BTC_Return'] > 0.00, 1, 0)
  df02.sample(10)

  # Shift the values in the "Trend" column
  # df02['Trend'] = df02['Trend'].shift(-1)

  # Get the last index value
  # last_index = df02.index[-1]
  # print(last_index)
  # Update the last value in the "Trend" column to NaN
  # df02.at[last_index, 'Trend'] = pd.NA
  # df02

  df02.sample(10)

  # Count the number of occurrences of Trend values
  trend_counts = df02['Trend'].value_counts()
  # Access the count for Trend = 0 and Trend = 1
  trend_0_count = trend_counts.get(0, 0)
  trend_1_count = trend_counts.get(1, 0)
  # Print the counts
  print("Count of Trend = 0:", trend_0_count)
  print("Count of Trend = 1:", trend_1_count)

  # Tratamiento de Missing Values (Valores Pérdidos):

  df03 = df02.dropna() # Eliminar filas con valores nulos

  # Particionamiento en Entrenamiento y Test:

  # Dividir los datos en conjunto de entrenamiento y prueba
  train_size = int(len(df03) * 0.8)
  train_data = df03[:train_size]
  test_data = df03[train_size:]

  print(df03.shape)
  print(train_size)
  print(train_data.shape)
  print(test_data.shape)

  # Normalización a la Escala entre 0 y 1 de los inputs:

  # Escalar los datos
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_train_data = scaler.fit_transform(train_data.drop(['BTC_Return'], axis=1))
  scaled_test_data = scaler.transform(test_data.drop(['BTC_Return'], axis=1))
  print(type(scaled_train_data))
  print(len(scaled_train_data))
  print(scaled_train_data.shape)

  # Crear secuencias de tiempo para el modelo LSTM
  window_size = 100

  def create_sequences(data):
    x = []
    y = []
    for i in range( window_size , len(data) ):
      x.append( data[i-window_size:i] )
      print("interacion: ", i-window_size +1 )
      print("indicador: ", i)
      print("posicion inicial tomada: ", i-window_size)
      print("posicion final tomada: ", i-1)
      print(data[i])
      # print(data[i-window_size:i])
      y.append( data[i][-1] )
      print("y[",i,"]: ", data[i][-1])

    return np.array(x), np.array(y)

  x_train, y_train = create_sequences(scaled_train_data)
  x_test, y_test = create_sequences(scaled_test_data)

  # Construir el modelo LSTM
  model = Sequential()
  model.add(LSTM(units=160, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

  # model.add(Dropout(0.2))
  model.add(LSTM(units=160))
  # model.add(Dropout(0.2))
  model.add(Dense(units=1, activation='sigmoid'))
  model.summary()

  # Compilar el modelo
  model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'])

  from IPython.display import SVG
  from keras.utils.vis_utils import model_to_dot

  SVG(model_to_dot(
    model,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=70,
    subgraph=False).create(prog='dot', format='svg'))

  # Entrenar el modelo
  model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1)
