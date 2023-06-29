#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout

import yfinance as yf

# Set the stock symbol and date range
symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2022-12-31"

# Download the historical stock data
data = yf.download(symbol, start=start_date, end=end_date)

# Save the data to a CSV file
data.to_csv("stock_data.csv")

data.head()

# #Loading the historical stock data
data = pd.read_csv("stock_data.csv")

#Preprocess the data
# Converting the 'date' column to datetime format
data['Date']=pd.to_datetime(data['Date'])

# Sort the data by date
data = data.sort_values('Date')

# Normalizing the 'close' prices between 0 and 1
scaler = MinMaxScaler(feature_range =(0,1))
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# split the data into training and testing sets
train_size = int(len(data)*0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

#preparing data for LSTM model
#create the training and testing data
def create_dataset(data, lookback):
    X,Y =[],[]
    for i in range(len(data)- lookback):
        X.append(data[i:i+lookback])
        Y.append(data[i+lookback])
    return np.array(X),np.array(Y)

#Set the lookback window size
lookback=25

# Create the training and testing datasets
train_X, train_Y = create_dataset(train_data['Close'].values, lookback)
test_X, test_Y = create_dataset(test_data['Close'].values, lookback)

# Reshape the input data for LSTM (samples, time steps, features)
train_X = np.reshape(train_X, (train_X.shape[0],train_X.shape[1],1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1],1))

#Building and training the LSTM model
# Creating the LSTM model
model = Sequential()
model.add(LSTM(units=50,return_sequences=True, input_shape=(lookback,1)))
model.add(Dropout(0.2))
model.add(LSTM(units =50))
model.add(Dropout(0.2))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam',loss='mean_squared_error')

#Train the model
model.fit(train_X, train_Y, epochs =10, batch_size =32)

#Make predictions with the LSTM model
#make predictions on the test data
predictions = model.predict(test_X)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(test_Y.reshape(-1,1))


# Plot the predicted and actual stock prices
plt.plot(actual_values,label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()