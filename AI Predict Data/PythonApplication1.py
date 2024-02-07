import streamlit as st
import pandas as pd
import numpy as np
from finta import TA
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Funcție pentru a obține datele de antrenament folosind yfinance
def get_training_data(symbol, api_key):
    stock_data = yf.Ticker(symbol)
    history_data = stock_data.history(period='1y', interval='1d')
    return history_data

# Definirea variabilelor symbol și api_key
symbol = "TSLA"
api_key = "your_api_key_here"  # înlocuiește cu cheia ta API

# Facem cererea API pentru a obține datele în timp real folosind yfinance
interval_yfinance = "30m"
data_yfinance = yf.download(symbol, interval=interval_yfinance)

# Transformăm datele într-un DataFrame Pandas
data_df_yfinance = pd.DataFrame(data_yfinance)

# Calculăm statistici de bază pentru dataframe
statistics_yfinance = data_df_yfinance.describe()

# Calculăm SMA și EMA cu ajutorul bibliotecii Finta
data_df_yfinance['SMA'] = TA.SMA(data_df_yfinance, 20)
data_df_yfinance['EMA'] = TA.EMA(data_df_yfinance, 20)

# Calculăm RSI cu ajutorul bibliotecii Finta
data_df_yfinance['RSI'] = TA.RSI(data_df_yfinance, 14)

# Obținem datele de antrenament
data_df_train = get_training_data(symbol, api_key)

# Antrenăm un model de regresie liniară pe datele de antrenament
X_train = data_df_train['Volume'].values.reshape(-1, 1)
y_train = data_df_train['Close'].values.reshape(-1, 1)
regression_model_train = LinearRegression()
regression_model_train.fit(X_train, y_train)

data_df = pd.DataFrame()  

# Obținem direcția tendinței de preț
last_close = data_df['Close'].iloc[-1] if 'Close' in data_df.columns else None

# Filtrăm datele istorice pentru următoarele 24 de ore
filtered_data_df = data_df.tail(24) if 'Close' in data_df.columns else pd.DataFrame()

# Obținem direcția tendinței de preț
last_close = data_df['Close'].iloc[-1]
current_price = data_df['Close'].iloc[-1]

# Filtrăm datele istorice pentru următoarele 24 de ore
filtered_data_df = data_df.tail(24)  # <-- Adaugă această linie

# Antrenăm filtered_regression_model pe datele filtrate
filtered_X_train = filtered_data_df['Volume'].values.reshape(-1, 1)
filtered_y_train = filtered_data_df['Close'].values.reshape(-1, 1)
filtered_regression_model = LinearRegression()
filtered_regression_model.fit(filtered_X_train, filtered_y_train)

# Folosim filtered_regression_model pentru a face predicții
filtered_predictions = filtered_regression_model.predict(filtered_X_train)
predicted_price = filtered_predictions[-1][0]
direction = "Cumpărare" if current_price < predicted_price else "Vânzare"

# Calculăm predicția concretă a prețului în moment și prețul concret al predicției
current_price_prediction = filtered_predictions[-1][0]
current_price_concrete_prediction = current_price_prediction + current_price

# Filtrăm datele istorice pentru următoarele 24 de ore
filtered_data_df = data_df.tail(24)  # Preluăm ultimele 24 de înregistrări (corespunzătoare ultimelor 24 de ore)

# Antrenăm un nou model de regresie liniară pe datele filtrate
filtered_X = filtered_data_df['Volume'].values.reshape(-1, 1)
filtered_y = filtered_data_df['Close'].values.reshape(-1, 1)
filtered_regression_model = LinearRegression()
filtered_regression_model.fit(filtered_X, filtered_y)
filtered_predictions = filtered_regression_model.predict(filtered_X)

# Obținem direcția tendinței de preț pentru intervalul specificat
filtered_last_close = filtered_data_df['Close'].iloc[-1]
filtered_current_price = filtered_data_df['Close'].iloc[-1]
filtered_predicted_price = filtered_predictions[-1][0]
filtered_direction = "Cumpărare" if filtered_current_price < filtered_predicted_price else "Vânzare"

# Configurăm interfața Streamlit
st.title('Analiza datelor cu yfinance')
st.subheader('Date procesate')

# Afișăm dataframe-ul cu datele procesate, inclusiv noii indicatori tehnici
st.write(data_df_yfinance)

# Afișăm statistici de bază
st.subheader('Statistici de bază')
st.write(statistics_yfinance)

# Afișăm graficul cu predicțiile regresiei liniare
st.subheader('Predicții regresie liniară')
st.line_chart(filtered_predictions)  # <-- Schimbă această linie

# Afișăm indicațiile concrete de cumpărare/vânzare și timpul estimat
st.subheader('Indicații de tranzacționare pentru perioada completă')
st.write(f"Ultimul preț de închidere: {last_close}")
st.write(f"Prețul curent: {current_price}")
st.write(f"Prețul estimat: {predicted_price:.2f}")
st.write(f"Direcția tendinței: {direction}")
st.write(f"Prețul estimat în moment: {current_price_prediction:.2f}")
st.write(f"Prețul concret al predicției: {current_price_concrete_prediction:.2f}")

# Afișăm dataframe-ul filtrat și statistici de bază pentru perioada specificată
st.subheader(f'Date procesate pentru următoarele 24 de ore')
st.write(filtered_data_df)

# Afișăm graficul cu predicțiile regresiei liniare pentru perioada specificată
st.subheader(f'Predicții regresie liniară pentru următoarele 24 de ore')
st.line_chart(filtered_predictions)

# Afișăm indicații concrete de cumpărare/vânzare pentru perioada specificată
st.subheader(f'Indicații de tranzacționare pentru următoarele 24 de ore')
st.write(f"Prețul curent: {filtered_current_price}")
st.write(f"Prețul estimat: {filtered_predicted_price:.2f}")
st.write(f"Direcția tendinței: {filtered_direction}")
