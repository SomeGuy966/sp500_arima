'''
README:
STEP #2

This file trains the ARIMA model based on the 'sp_parsed.csv'
data.

It then saves this model to the 'arima_model.pkl' pickle file.
'''



def data_download(start_date, end_date):
    # Downloads S&P 500 data
    import yfinance as yf
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import requests

    #today = datetime.now()
    #start = (today - relativedelta(years=50)).strftime('%Y-%m-%d')
    #end = today.strftime('%Y-%m-%d')

    sp500 = yf.download("^GSPC", start=start_date, end=end_date)

    # DataFrame column names have two levels: "Price" and "Ticker." Drop the levels, keep the column names
    sp500.columns = sp500.columns.droplevel("Ticker")
    sp500.columns.name = None
def augmented_dickey_fuller_test(time_series_data):
    # Applying the Augmented Dickey-Fuller test to check for stationarity in the time series
    from statsmodels.tsa.stattools import adfuller

    significance_level = 0.05
    check = adfuller(time_series_data)

    ADF_statistic, p_value,_,_,_,_ = check

    print(f"ADF statistic: {ADF_statistic}")
    print(f"p-value: {p_value}")

    if p_value > significance_level:
        print("FAILED: p-value above significance level\n")
        print("Differencing the time series...\n\n")

        return time_series_data.diff()
    else:
        print("PASSED: p-value below significance level\n")


# Step 1: Determining best parameters for the ARIMA model
import pmdarima as pm
import pandas as pd
from pmdarima.model_selection import train_test_split

sp500_data = pd.read_csv('sp_parsed.csv')# Reading the data; sets 1st column (Dates) as index
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
sp500_data.set_index('Date', inplace=True)


y_train, y_test = train_test_split(sp500_data['Close'], train_size=0.8)  # Splits the data into 80% train/20% test


model_auto = pm.auto_arima(y_train,                # Input training data
                           start_p=0,              # Starting autoregressive order
                           start_q=0,              # Starting moving average order
                           max_p=5,                # Max autoregressive order
                           max_q=5,                # Max moving average order
                           d=1,                    # Number of differencing steps
                           seasonal=False,         # Data doesn't have patterns repeated over time
                           trace=True,             # Prints model parameters while training
                           error_action='ignore',  # When model encounters error, "skips" the error
                           suppress_warnings=True, # When warnings appear while training, they're ignored
                           random_state=0,
                           n_fits=50
                            )



optimal_order = model_auto.order   # Accessing .order attribute to find optimal order
p, d, q = optimal_order            # Unpacking the 3-tuple to get best parameters




# Step 2: Training the model
from statsmodels.tsa.arima.model import ARIMA

model_generated = ARIMA(y_train, order=(p, d, q))
model = model_generated.fit()
model.summary()



# Step 3: Saving the trained model
import pickle

with open("arima_model.pkl", "wb") as f:
    pickle.dump(model, f)


print('\n\nModel saved to "arima_model.pkl"!')
