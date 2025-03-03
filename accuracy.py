'''
README:
This program calculates the mean average percent error (MAPE)
of the model. The model forecasts one month at a time for every month
between 1975-2025 using one year’s worth of data prior to that month
to train the model.

Then it calculates MAPE for the forecasts vs. the real prices, and
saves the calculated monthly MAPE to a .csv file called "monthly_MAPE.csv"
'''


import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_percentage_error
from model_training import train_model
import pytz



def months_between(date1: str, date2: str, date_format="%Y-%m-%d") -> int:
    d1 = datetime.strptime(date1, date_format)
    d2 = datetime.strptime(date2, date_format)

    # Calculate the difference in months
    total_months = (d2.year - d1.year) * 12 + (d2.month - d1.month)

    # Add 1 to count the starting month as well as the ending month
    return abs(total_months)


sp500 = pd.read_csv('sp500_data.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'], utc=True)
sp500.set_index('Date', inplace=True)


# Define your dates
start_date = "1975-01-01"
end_date = "2025-01-01"
m = months_between(start_date, end_date)

start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

error = 0


for i in range(0, m):
    monthly_data = sp500[(sp500.index.month == start_date.month) & (sp500.index.year == start_date.year)]
    monthly_data = monthly_data['Close']

    filtered_data = sp500[
        (sp500.index >= (start_date - relativedelta(years=1))) &
        (sp500.index < start_date)
        ]
    filtered_data = filtered_data['Close']


    model = train_model(filtered_data)

    forecast = pd.DataFrame(model.forecast(monthly_data.shape[0]))
    forecast['Date'] = monthly_data.index
    forecast = forecast.set_index('Date')
    forecast.columns = ['Predictions']

    monthly_error = mean_absolute_percentage_error(monthly_data, forecast['Predictions']) * 100
    error += monthly_error

    formatted_date = start_date.strftime("%B %Y")
    print(f"MAPE for {formatted_date}: {monthly_error}")

    start_date += relativedelta(months=1)


error = error/m
print(f"Monthly MAPE from 1975-2025: {error}")


df = pd.DataFrame({
	'MAPE': [error]
})

df.to_csv('monthly_MAPE.csv', index=False)


