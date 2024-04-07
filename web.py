import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Price Prediction')
company = st.text_input('Enter Stock ID', 'GOOG')

end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

company_data = yf.download(company, start=start, end=end)
model = load_model('model.h5')
st.subheader("Stock Data")
st.write(company_data)

splitting_len = int(len(company_data)*0.8)
x_test = pd.DataFrame(company_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, "Orange")
    plt.plot(full_data.Close, "b")
    if extra_data:
        plt.plot(extra_dataset)
    return fig

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(np.array(x_test[['Close']]))

x_data = []
y_data = []

for i in range(60,len(scaled_data)):
    x_data.append(scaled_data[i-60:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
{
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
},
        index=company_data.index[splitting_len+60:]
)
st.subheader("Original values vs predicted values")
st.write(ploting_data)

st.subheader("Original Close Price and predicted Close Price")
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([company_data.Close[:splitting_len+60], ploting_data], axis=0))
plt.legend(["Data-not-usesd", "Original-Close-Price", "Predicted-test-Price"])
st.pyplot(fig)

