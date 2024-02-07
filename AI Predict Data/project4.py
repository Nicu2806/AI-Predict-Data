import streamlit as st
import yfinance as yf
from finta import TA

# Funcție pentru a obține datele de la bursa folosind yfinance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Funcție pentru a calcula indicatorii SMA, EMA și RSI
def calculate_indicators(data):
    data['SMA'] = TA.SMA(data, 20)
    data['EMA'] = TA.EMA(data, 20)
    data['RSI'] = TA.RSI(data, 14)
    return data

# Configurare Streamlit
st.title('Analiza Tehnică a Acțiunilor')
st.sidebar.header('Setări')

# Selectare simbol
symbol = st.sidebar.text_input('Introduceți simbolul acțiunii:', 'AAPL')

# Selectare interval de timp
start_date = st.sidebar.text_input('Data de început (YYYY-MM-DD):', '2022-01-01')
end_date = st.sidebar.text_input('Data de sfârșit (YYYY-MM-DD):', '2023-01-01')

# Obținere date de la bursa
stock_data = get_stock_data(symbol, start_date, end_date)

# Calculare indicatori
stock_data = calculate_indicators(stock_data)

# Afișare date brute
st.subheader('Date brute de la bursa')
st.write(stock_data)

# Afișare grafic pentru SMA, EMA și RSI
st.subheader('Grafic SMA, EMA și RSI')
st.line_chart(stock_data[['Close', 'SMA', 'EMA', 'RSI']])

# Afișare statistici
st.subheader('Statistici de bază')
st.write(stock_data.describe())
