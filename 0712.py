import pandas as pd
from datetime import time

# Load data
df = pd.read_csv('c:/sqlite/CME-NQ/USATECH.IDXUSD_Candlestick_12_M_BID_16.05.2022-15.05.2025.csv')

# Parse datetime and convert to CST
df['datetime'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
df['datetime_cst'] = df['datetime'] - pd.Timedelta(hours=6)
df['date_cst'] = df['datetime_cst'].dt.date
df['time_cst'] = df['datetime_cst'].dt.time

# Define time windows
range_start = time(7, 12)
range_end = time(8, 0)
test_start = time(8, 0)
test_end = time(12, 0)

results = []
range_results = []
traded_results = []

for day, group in df.groupby('date_cst'):
    # Get range window
    range_window = group[(group['time_cst'] >= range_start) & (group['time_cst'] <= range_end)]
    if str(day) == "2025-05-06":
        print(f"Range window for {day}:")
        print(range_window[['datetime_cst', 'High', 'Low']])
    if range_window.empty:
        continue
    range_high = range_window['High'].max()
    range_low = range_window['Low'].min()
    range_results.append((day, range_high, range_low))
    
    # Get test window
    test_window = group[(group['time_cst'] >= test_start) & (group['time_cst'] <= test_end)]
    if test_window.empty:
        continue
    
    # Check if high or low are traded through
    high_traded = (test_window['High'] > range_high).any()
    low_traded = (test_window['Low'] < range_low).any()
    traded_results.append((day, high_traded, low_traded))

# Print latest 10 results
print("Latest 10 results for 07:12CST to 08:00CST range and if high/low traded through 08:00-12:00 CST:")
for r, t in zip(range_results[-10:], traded_results[-10:]):
    print(f"Date: {r[0]}, High: {r[1]}, Low: {r[2]}, High Traded: {t[1]}, Low Traded: {t[2]}")