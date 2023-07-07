import streamlit as st

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf

def app():
    st.title('Modelo SVC')

    start_date = '2015-01-01'
    # Set start and end dates for the price data
    # Establecer fechas de inicio y finalización para los datos de precios
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    df = yf.download('BTC-USD', start=start_date, end=end_date)
    df

    # Changes The Date column as index columns
    # df.index = pd.to_datetime(df['Date'])

    # drop The original date column
    # df = df.drop(['Date'], axis='columns')

    # Create predictor variables
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
    X.head()

    # Target variables
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    y

    split_percentage = 0.8
    split = int(split_percentage*len(df))
    
    # Train data set
    X_train = X[:split]
    y_train = y[:split]

    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    # Support vector classifier
    cls = SVC().fit(X_train, y_train)

    df['Predicted_Signal'] = cls.predict(X)
    df['Predicted_Signal']

    # Calculate daily returns
    df['Return'] = df.Close.pct_change()
    df['Return']

    # Calculate strategy returns
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)
    df['Strategy_Return']

    # Calculate Cumulutive returns
    df['Cum_Ret'] = df['Return'].cumsum()
    df

    # # Plot Strategy Cumulative returns
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    df

    plt.plot(df['Cum_Ret'],color='red')
    plt.plot(df['Cum_Strategy'],color='blue')
