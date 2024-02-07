import streamlit as st
import yfinance as yf
from finta import TA
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#

# Examinam stocurile
aapl_info['Volume'].plot(figsize=(10, 7))
plt.title('AAPL Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

# Descarcam data pentru analiza
data = yf.download(['AAPL', 'FSLR', 'NVDA', 'ARKK', 'SPY'], period='1y')['Close']
correlation_matrix = data.corr()
print(correlation_matrix)

# Declaram scalarea 
aapl_info = aapl_info.fillna(aapl_info.mean()) 
aapl_info['MA_5'] = aapl_info['Close'].rolling(window=5).mean()  
scaler = MinMaxScaler(feature_range=(0,1))
aapl_info_scaled = scaler.fit_transform(aapl_info)

# Declaram axele regresiei liniare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Încărcare model și tokenizator GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Funcție pentru generarea de texte cu modelul GPT-2
def generate_text(prompt, max_length=100, num_return_sequences=1):
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

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

# Functie pentru formarea predictiilor 
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 60
X_train, y_train = create_dataset(X_train, y_train, TIME_STEPS)
X_test, y_test = create_dataset(X_test, y_test, TIME_STEPS)

# Configurare Streamlit
st.title('Technical Analysis of Stocks')
st.sidebar.header('Setings')

# Selectare simbol
symbol = st.sidebar.text_input('Enter the action symbol:', 'AAPL')

# Selectare interval de timp
start_date = st.sidebar.text_input('Start date (YYYY-MM-DD):', '2023-01-01')
end_date = st.sidebar.text_input('End date (YYYY-MM-DD):', '2023-11-16')

# Obținere date de la bursa
stock_data = get_stock_data(symbol, start_date, end_date)

# Calculare indicatori
stock_data = calculate_indicators(stock_data)

# Afișare date brute
st.subheader('Raw data from the stock market')
st.write(stock_data)

# Afișare grafic pentru SMA, EMA și RSI
st.subheader('Grafic SMA, EMA și RSI')
st.line_chart(stock_data[['Close', 'SMA', 'EMA', 'RSI']])

# Afișare statistici
st.subheader('Basic statistics')
st.write(stock_data.describe())

# Introducere întrebare
question = st.text_input('Enter a question about the stock market:', 'What is happening to the stock market today?')

# Afișare răspuns generat de GPT-2
st.subheader('Answer to the question')
generated_text = generate_text(question)
st.write(generated_text)

