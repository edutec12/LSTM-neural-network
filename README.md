# LSTM-neural-network
This code is an implementation of an LSTM neural network for making time series predictions. The goal is to predict the next values of a time series using a machine learning model. The time series is defined as a sequence of values in a variable over time, and in this case, the data provided in the "data" array is used as the time series.

The code first loads the input data and processes it for subsequent training and testing. Then, the data is normalized to be in a uniform range, and the input data is split into training and testing sets.

Next, the structure of the LSTM neural network is defined, which consists of an input layer, an LSTM layer, a fully connected layer, and a regression layer. The network is trained using the training data, and predictions are made for the next 20 values of the time series.

![pronostico](https://user-images.githubusercontent.com/97995445/219577041-f884f560-e349-4424-9f6f-dcb922659d77.png)
