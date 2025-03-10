import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo
import time

# Start timing the execution
start_time = time.time()

print("Simple Analysis: Trading Through Early Hour Highs")
print("=" * 60)

# Find data files
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Look for the 1-minute data file
data_paths = [
    os.path.join(base_dir, 'nq-ohlcv-1m.csv'),
    os.path.join(parent_dir, 'nq-ohlcv-1m.csv'),
    r'c:\sqlite\nq-ohlcv-1m.csv'
]

# Find the first available path
data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("Error: Could not find the 1-minute data file.")
    sys.exit(1)

print(f"Using data file: {data_path}")

# Load the data file with only necessary columns to reduce memory usage
print("Loading data...")
df = pd.read_csv(data_path, usecols=['ts_event', 'symbol', 'high', 'low', 'volume'])

# Filter out spread symbols (containing hyphens)
print("Filtering out spread symbols...")
df = df[~df['symbol'].str.contains('-')]

# Filter out rows with zero volume
df = df[df['volume'] > 0]

print(f"Data after filtering: {len(df):,} rows")

# Convert timestamp to datetime
print("Processing timestamps...")
df['datetime_utc'] = pd.to_datetime(df['ts_event'], unit='ns')

# Convert to Chicago time
chicago_tz = ZoneInfo('America/Chicago')
df['datetime'] = df['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert(chicago_tz).dt.tz_localize(None)

# Extract date, hour, minute for easier grouping
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['day_of_week'] = df['datetime'].dt.dayofweek + 1  # 1=Monday, 7=Sunday

# Filter out weekends
df = df[df['day_of_week'] < 6]

# Add hour start time for grouping
df['hour_start'] = df['datetime'].dt.floor('h')
df['next_hour_start'] = df['hour_start'] + timedelta(hours=1)

# Create unique hour identifiers
df['hour_id'] = df['symbol'] + '_' + df['hour_start'].dt.strftime('%Y-%m-%d_%H')
df['next_hour_id'] = df['symbol'] + '_' + df['next_hour_start'].dt.strftime('%Y-%m-%d_%H')

print("Analyzing price patterns...")

# Process in batches by symbol to reduce memory usage
symbols = df['symbol'].unique()
print(f"Analyzing {len(symbols)} unique symbols...")
results = []

for symbol in symbols:
    print(f"Analyzing {symbol}...")
    symbol_data = df[df['symbol'] == symbol].copy()
    
    # Group by hour
    for hour_id, hour_group in symbol_data.groupby('hour_id'):
        # Skip if this hour has insufficient data
        if len(hour_group) < 45:  # Allow for some missing data
            continue
            
        # Get the hour start time
        hour_start = hour_group['hour_start'].iloc[0]
        next_hour_id = hour_group['next_hour_id'].iloc[0]
        
        # Identify the first 12 minutes
        first_12_min = hour_group[hour_group['minute'] < 12]
        if len(first_12_min) < 10:  # Need at least 10 minutes of data
            continue
            
        # Get the high of the current hour and first 12 minutes
        hour_high = hour_group['high'].max()
        first_12_high = first_12_min['high'].max()
        
        # Check if high was set in first 12 minutes
        high_in_first_12 = abs(first_12_high - hour_high) < 0.01
        
        if not high_in_first_12:
            continue  # Skip if high wasn't in first 12 min
            
        # Get data for the next hour
        next_hour_data = symbol_data[symbol_data['hour_id'] == next_hour_id]
        
        # Skip if next hour doesn't have enough data
        if len(next_hour_data) < 30:
            continue
            
        # Check if price trades above the first 12 min high in the next hour
        next_hour_high = next_hour_data['high'].max()
        trades_above_high = next_hour_high > first_12_high
        
        # Calculate how much the price exceeds the early high (when it does)
        if trades_above_high:
            penetration = next_hour_high - first_12_high
        else:
            penetration = 0
        
        results.append({
            'symbol': symbol,
            'hour_start': hour_start,
            'hour': hour_start.hour,
            'trades_above_high': trades_above_high,
            'early_high': first_12_high,
            'next_hour_high': next_hour_high,
            'penetration': penetration
        })
        
    # Clean up to save memory
    del symbol_data

# Convert results to DataFrame
print("Compiling results...")
results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("No valid data found for analysis. Please check your data file.")
    sys.exit(1)

# Calculate overall probability
overall_prob = results_df['trades_above_high'].mean() * 100
avg_penetration = results_df[results_df['trades_above_high']]['penetration'].mean()

print("\nRESULTS")
print("=" * 60)
print(f"Total hours analyzed where high was set in first 12 minutes: {len(results_df):,}")
print(f"Probability of trading above the early high in next hour: {overall_prob:.2f}%")
print(f"When broken, average penetration above the high: {avg_penetration:.2f} points")

# Calculate probability by hour of day
by_hour = results_df.groupby('hour').agg({
    'trades_above_high': 'mean',
    'penetration': 'mean',
    'symbol': 'count'
}).reset_index()
by_hour['trades_above_high'] *= 100
by_hour.rename(columns={'trades_above_high': 'probability', 
                        'penetration': 'avg_penetration', 
                        'symbol': 'samples'}, inplace=True)

# Add market context
def get_market_context(hour):
    if hour == 8:
        return "CME Open (8:30 CT)"
    elif hour == 15:
        return "CME Close (15:15 CT)"
    elif 8 <= hour <= 15:
        return "Regular Trading"
    elif 16 <= hour <= 17:
        return "Globex Reopen"
    else:
        return "Extended Hours"

by_hour['context'] = by_hour['hour'].apply(get_market_context)

# Print results by hour
print("\nPROBABILITY BY HOUR")
print("=" * 100)
print("{:<6} {:<15} {:<20} {:<10} {:<20}".format(
    "Hour", "Probability %", "Avg Penetration", "Samples", "Market Context"))
print("-" * 100)

for _, row in by_hour.iterrows():
    print("{:<6} {:<15.2f} {:<20.2f} {:<10} {:<20}".format(
        f"{int(row['hour']):02d}:00", 
        row['probability'],
        row['avg_penetration'],
        row['samples'],
        row['context']
    ))

# Calculate average during regular vs extended hours
regular_hours = by_hour[(by_hour['hour'] >= 8) & (by_hour['hour'] <= 15)]
extended_hours = by_hour[(by_hour['hour'] < 8) | (by_hour['hour'] > 15)]

reg_avg = regular_hours['probability'].mean() if len(regular_hours) > 0 else 0
ext_avg = extended_hours['probability'].mean() if len(extended_hours) > 0 else 0

reg_samples = regular_hours['samples'].sum() if len(regular_hours) > 0 else 0
ext_samples = extended_hours['samples'].sum() if len(extended_hours) > 0 else 0

print("\nSESSION COMPARISON")
print("=" * 80)
print(f"Regular Trading Hours (8:00-15:00 CT): {reg_avg:.2f}% (samples: {reg_samples:,})")
print(f"Extended Trading Hours: {ext_avg:.2f}% (samples: {ext_samples:,})")

# Execution time
end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")