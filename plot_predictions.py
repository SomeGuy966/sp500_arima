'''
README:
STEP #3

This file takes the model trained in step #2 ('arima_model.pkl')
and uses it to predict the last 20% of the 'sp_parsed.csv' data.

It then plots the training data (first 80%), test data (last 20%),
the predictions, and the range that the model predicts the test data
could fall into.
'''


import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Importing model
with open('arima_model.pkl', 'rb') as f:
    model = pickle.load(f)

sp500_data = pd.read_csv('sp_parsed.csv')
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
sp500_data.set_index('Date', inplace=True)

y_train, y_test = train_test_split(sp500_data['Close'], train_size=0.8)


# Obtain predictions from ARMA(1,0) model on X_t
ar_preds = pd.DataFrame(model.forecast(y_test.shape[0]))
ar_preds['Date'] = y_test.index
ar_preds = ar_preds.set_index('Date')
ar_preds.columns = ['predictions']

# Get CI
ar_pred_ci = model.get_forecast(y_test.shape[0]).conf_int()
ar_pred_ci['Date'] = y_test.index
ar_pred_ci = ar_pred_ci.set_index('Date')
# Make plot of predictions
plt.plot(y_train,label="Training")
plt.plot(y_test,label="Testing")
plt.plot(ar_preds['predictions'],label="Predicted")
plt.fill_between(ar_pred_ci.index, ar_pred_ci.iloc[:,0], ar_pred_ci.iloc[:,1], color='blue', alpha=0.3)
plt.legend(loc = 'upper left')


p, d, q = model.model.order


plt.title(f'Predictions using ARIMA({p},{d},{q})')
plt.show()


data = {'MAPE': [mean_absolute_percentage_error(y_test,ar_preds['predictions']) * 100],
        'MSE': [mean_squared_error(y_test,ar_preds['predictions'])]}

ar_preds.to_csv('predictions.csv')
print(f'\nSaved to "predictions.csv!"\n')

df = pd.DataFrame(data, index = ["ARIMA"])
print(df.head())





