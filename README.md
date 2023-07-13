# Stock-Price-Prediction
## Introduction:
This documentation describes a stock price prediction model implemented in Python to forecast the market value of Tesla stock for the next 30 days. The model utilizes historical stock price data and employs a Long Short-Term Memory (LSTM) neural network to make predictions. The project involves data preprocessing, model building, hyperparameter tuning, evaluation, and visualization of the predicted stock prices.

## Data Collection:

Importing Libraries: The necessary libraries, including pandas, pandas_datareader, and os, are imported.

Data Collection: The stock price data for Tesla (TSLA) is collected using the Tiingo API and stored in a CSV file named 'TESLA.csv'.

## Data Preprocessing:

Loading the Dataset: The dataset is loaded using the pandas library from the 'TESLA.csv' file.

Exploratory Data Analysis (EDA): The dataset is analyzed by displaying the head, tail, and basic statistics of the closing stock prices. Additionally, line plots are created to visualize the historical stock price trends.

Feature Scaling: MinMaxScaler from scikit-learn is applied to scale the closing stock prices between 0 and 1. This scaling is necessary for the LSTM model to handle the data effectively.

## Model Building:

Data Splitting: The scaled data is split into training and testing sets. The training set contains 75% of the data, while the testing set contains the remaining 25%.

Time Series Data Preparation: The data is reshaped into a suitable format for the LSTM model. A function is created to split the data into input (x) and output (y) sequences with a specified time step. The training and testing datasets are transformed accordingly.

LSTM Model Creation: A stacked LSTM model is built using the Sequential model from the Keras library. The model consists of multiple LSTM layers followed by a dense output layer. The model is compiled using the mean squared error (MSE) loss function and the Adam optimizer.

Hyperparameter Tuning: The model is trained on the training data with a specified number of epochs and a batch size. The loss values during training are plotted to identify the optimal number of epochs that minimize both training and validation loss.

Prediction and Evaluation: The trained model is used to predict the stock prices for both the training and testing datasets. The predicted values are inverse transformed to obtain the original scale. The root mean squared error (RMSE) and mean absolute error (MAE) are calculated to evaluate the model's performance.

## Visualization:

Plotting Baseline and Predictions: Line plots are created to visualize the original stock prices, the predicted stock prices for the training set, and the predicted stock prices for the testing set.

Forecasting Future Stock Prices: The model is used to forecast the stock prices for the next 30 days. The predicted values are appended to the original dataset, and a line plot is created to display the forecasted stock prices along with the historical data.

## Conclusion:
This documentation outlines the implementation of a stock price prediction model for Tesla stock using LSTM. The model effectively utilizes historical stock price data to forecast the market value for the next 30 days. By evaluating the model's performance metrics such as RMSE and MAE, users can gain insights into the model's accuracy in predicting stock prices. The visualizations provide a clear representation of the historical and forecasted stock prices, aiding in decision-making for investors and financial professionals.
