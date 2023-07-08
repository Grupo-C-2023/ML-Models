import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

def app():
    plt.style.use('fivethirtyeight')

    # Setting figure size  
    rcParams['figure.figsize'] = 20, 10

    #for normalizing data
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fetching data from Yahoo Finance
    ticker = 'DIS'
    start = st.date_input('Start Train' , value=pd.to_datetime('2018-1-1'))
    end = st.date_input('End Train' , value=pd.to_datetime('today'))
    period1 = int(time.mktime(start.timetuple()))
    period2 = int(time.mktime(end.timetuple()))
    interval = '1d'  # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)

    df['symbol'] = 'DIS'
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(['Date'], axis='columns')
    
    st.subheader("Original Data")
    st.write(df)

    df = df[['Close']]

    st.subheader("Close Prices")
    st.write(df)

    # Create a variable to predict 'x' days out in the future
    future_days = 100
    # Create a new column (target) shifted 'x' units/days up
    df['Prediction'] = df[['Close']].shift(-future_days)
    
    st.subheader("Data with Prediction Column")
    st.write(df.tail(10))

    X = np.array(df.drop(['Prediction'], axis=1))[:-future_days]
    y = np.array(df['Prediction'])[:-future_days]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Fit the decision tree regressor
    tree = DecisionTreeRegressor().fit(x_train, y_train)

    x_future = df.drop(['Prediction'], axis=1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    
    st.subheader("Future Data")
    st.write(x_future)

    # Make predictions on future data
    tree_prediction = tree.predict(x_future)
    
    st.subheader("Predictions")
    st.write(tree_prediction)

    rms = np.sqrt(np.mean(np.power((x_future - tree_prediction), 2)))
    
    st.subheader("Root Mean Squared Error")
    st.write(rms)

    # Visualize the data
    st.subheader("Visualizing the Data")
    predictions = tree_prediction

    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    fig = plt.figure(figsize=(16, 8))
    plt.title('Model Decision Tree')
    plt.xlabel('Days')
    plt.ylabel('Close Price USD($)')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Original Data', 'Valid Data', 'Predicted Data'])

    st.pyplot(fig)