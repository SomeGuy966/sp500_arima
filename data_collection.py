'''
README:
STEP #1

This file takes the 'sp500_data.csv' data and
parses it to only contain an interval of dates you select.

This is because the 'sp500_data.csv' data contains data
from 1993-2024. Since we're taking our data and predicting
for the last 20% of it, 31 years of data is too large a time-frame.

ARIMA models typically only work well for predicting
within the short-term, not the long-term.

The parsed data is saved to 'sp_parsed.csv'
'''


import pandas as pd
from datetime import datetime
from dateutil import parser
import pytz

data = pd.read_csv('sp500_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

start = '2024-01-01'
end = '2025-01-01'


start_date = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
end_date = datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=pytz.UTC)

if start_date > end_date:
    raise ValueError('Error: Start date is after end date')
elif start_date < data['Date'].iloc[0]:
    raise ValueError('Error: Start date is before earliest data')



filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

filtered_data.to_csv('sp_parsed.csv', index=False)

print(f"\n\nData from {start} to {end} successfully parsed!")
print("Saved to 'sp_parsed.csv!'")



