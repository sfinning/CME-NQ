import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo  # Modern timezone handling

start_time = time.time()
print("=" * 60)
print("Analyzing which 12-minute cycle creates the HIGH in BEARISH hours")
print("=" * 60)

# Load the 12-minute data
file_path = r"c:\sqlite\CME-NQ\nq-ohlcv-12m.csv"

print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)
print(f"Loaded {len(df):,} rows of 12-minute data")

# Convert timestamp to datetime
if 'ts_event' in df.columns:
    sample_value = df['ts_event'].iloc[0]
    unit = 'ns' if len(str(int(sample_value))) > 13 else 'ms'
    df['datetime'] = pd.to_datetime(df['ts_event'], unit=unit)
else:
    print("Error: Could not find timestamp column")
    exit(1)

# Convert UTC to Chicago time with proper timezone handling
print("Converting timestamps from UTC to Chicago time (CT)...")

# Define a function to convert UTC timestamps to Chicago time
def utc_to_chicago(dt):
    if pd.isna(dt):
        return pd.NaT
    # Localize to UTC then convert to Chicago time
    dt_utc = dt.replace(tzinfo=ZoneInfo('UTC'))
    dt_chicago = dt_utc.astimezone(ZoneInfo('America/Chicago'))
    return dt_chicago

# Apply the conversion to all timestamps
# This vectorized approach is more efficient than .apply()
df['chicago_time'] = pd.Series([utc_to_chicago(dt) for dt in df['datetime']])

# Create hour and 12-minute cycle columns from Chicago time
df['hour'] = df['chicago_time'].dt.hour
df['minute'] = df['chicago_time'].dt.minute
df['12min_cycle'] = (df['minute'] // 12) + 1  # 1 through 5
df['hour_start'] = df['chicago_time'].dt.floor('h')

print("Creating hourly candles...")
# Group by hour and symbol to create hourly candles
hourly_df = df.groupby(['hour_start', 'symbol']).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min', 
    'close': 'last'
}).reset_index()

# Add hour of day to hourly candles for analysis
hourly_df['hour_of_day'] = hourly_df['hour_start'].dt.hour

# Identify bearish hourly candles (DIFFERENT FROM ORIGINAL: close < open)
hourly_df['is_bearish'] = hourly_df['close'] < hourly_df['open']
bearish_count = hourly_df['is_bearish'].sum()
print(f"Created {len(hourly_df):,} hourly candles")
print(f"Found {bearish_count:,} bearish hours ({bearish_count/len(hourly_df)*100:.1f}%)")

# For each bearish hour, identify which 12-min cycle created the high
print("Analyzing which 12-minute cycle sets the high in bearish hours...")

# Initialize counters - one for each hour of day
cycle_counts_by_hour = {}
for h in range(24):
    cycle_counts_by_hour[h] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

# Overall counts
overall_cycle_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
processed_count = 0

# Create a lookup dictionary for hourly highs (DIFFERENT FROM ORIGINAL: highs instead of lows)
hourly_highs = {}
for _, row in hourly_df[hourly_df['is_bearish']].iterrows():
    key = (row['hour_start'], row['symbol'])
    hourly_highs[key] = (row['high'], row['hour_of_day'])

# Group data by hour_start and symbol
grouped = df.groupby(['hour_start', 'symbol'])

for (hour_start, symbol), group in grouped:
    # Check if this is a bearish hour
    key = (hour_start, symbol)
    if key not in hourly_highs:
        continue
        
    hourly_high, hour_of_day = hourly_highs[key]
    
    # Find the first 12-minute candle that has the high
    high_candles = group[group['high'] == hourly_high]
    if not high_candles.empty:
        cycle = high_candles.iloc[0]['12min_cycle']
        
        # Update both overall and hour-specific counters
        overall_cycle_counts[cycle] += 1
        cycle_counts_by_hour[hour_of_day][cycle] += 1
        
        processed_count += 1

print(f"Successfully analyzed {processed_count:,} bearish hours")

# Calculate overall percentages
total = sum(overall_cycle_counts.values())
overall_percentages = {cycle: (count / total * 100) for cycle, count in overall_cycle_counts.items()}

# Display overall results
print("\nOVERALL RESULTS (Chicago Time)")
print("=" * 50)
print("\nDistribution of 12-minute cycles creating the HIGH in bearish hourly candles:")
for cycle in sorted(overall_cycle_counts.keys()):
    count = overall_cycle_counts[cycle]
    percentage = overall_percentages[cycle]
    print(f"Cycle {cycle} (minutes {(cycle-1)*12}-{cycle*12-1}): {count:,} occurrences ({percentage:.1f}%)")

# Identify the cycle with the highest probability overall
max_cycle = max(overall_percentages, key=overall_percentages.get)
print(f"\nThe 12-minute cycle with the highest probability of creating the HIGH in a bearish hourly candle:")
print(f"Cycle {max_cycle} (minutes {(max_cycle-1)*12}-{max_cycle*12-1}) with {overall_percentages[max_cycle]:.1f}% probability")

# Display results for each hour of the day
print("\n\nBREAKDOWN BY HOUR OF DAY (Chicago Time)")
print("=" * 50)

# Create header for the table
header = "Hour (CT) | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Most Likely"
print(f"\n{header}")
print("-" * len(header))

# Process each hour
for hour in range(24):
    # Skip hours with no data
    hour_total = sum(cycle_counts_by_hour[hour].values())
    if hour_total == 0:
        continue
    
    # Calculate percentages for this hour
    hour_percentages = {cycle: (count / hour_total * 100) for cycle, count in cycle_counts_by_hour[hour].items()}
    
    # Find max for this hour
    max_cycle_hour = max(hour_percentages, key=hour_percentages.get)
    max_pct = hour_percentages[max_cycle_hour]
    
    # Format the row
    row = f"{hour:02d}:00 CT | "
    for cycle in range(1, 6):
        pct = hour_percentages.get(cycle, 0)
        if cycle == max_cycle_hour:
            row += f"*{pct:5.1f}%* | "  # Highlight max
        else:
            row += f" {pct:5.1f}%  | "
    
    row += f"Cycle {max_cycle_hour} ({max_pct:.1f}%)"
    print(row)

# CME Trading hours summary
print("\nCME TRADING HOURS SUMMARY (Chicago Time)")
print("=" * 50)

# Define CME NQ futures standard trading hours
# CME Globex hours for NQ: Sunday-Friday 5:00 PM to 4:00 PM CT the next day
cme_hours = list(range(17, 24)) + list(range(0, 16))  # 5PM to 4PM next day

# Combine data for CME trading hours
cme_hour_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for hour in cme_hours:
    for cycle in range(1, 6):
        cme_hour_counts[cycle] += cycle_counts_by_hour[hour].get(cycle, 0)

# Calculate percentages
cme_total = sum(cme_hour_counts.values())
if cme_total > 0:
    cme_percentages = {cycle: (count / cme_total * 100) for cycle, count in cme_hour_counts.items()}
    
    # Find max for CME trading hours
    max_cycle_cme = max(cme_percentages, key=cme_percentages.get)
    
    print("\nDistribution of 12-minute cycles creating the HIGH during CME trading hours:")
    for cycle in sorted(cme_hour_counts.keys()):
        count = cme_hour_counts[cycle]
        percentage = cme_percentages[cycle]
        print(f"Cycle {cycle} (minutes {(cycle-1)*12}-{cycle*12-1}): {count:,} occurrences ({percentage:.1f}%)")
    
    print(f"\nDuring CME trading hours, Cycle {max_cycle_cme} " +
          f"(minutes {(max_cycle_cme-1)*12}-{max_cycle_cme*12-1}) " +
          f"has the highest probability at {cme_percentages[max_cycle_cme]:.1f}%")
else:
    print("No data found for CME trading hours")

# Regular Market Hours summary (8:30 AM to 3:15 PM CT for NQ)
print("\nREGULAR MARKET HOURS SUMMARY (8:30 AM to 3:15 PM CT)")
print("=" * 50)

# Define Regular Market Hours for NQ futures
regular_hours = list(range(9, 16)) + [8]  # 8:30 AM to 3:15 PM (approximated to hourly data)

# Combine data for Regular Market Hours
regular_hour_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for hour in regular_hours:
    for cycle in range(1, 6):
        regular_hour_counts[cycle] += cycle_counts_by_hour[hour].get(cycle, 0)

# Calculate percentages
regular_total = sum(regular_hour_counts.values())
if regular_total > 0:
    regular_percentages = {cycle: (count / regular_total * 100) for cycle, count in regular_hour_counts.items()}
    
    # Find max for regular trading hours
    max_cycle_regular = max(regular_percentages, key=regular_percentages.get)
    
    print("\nDistribution of 12-minute cycles creating the HIGH during Regular Market Hours:")
    for cycle in sorted(regular_hour_counts.keys()):
        count = regular_hour_counts[cycle]
        percentage = regular_percentages[cycle]
        print(f"Cycle {cycle} (minutes {(cycle-1)*12}-{cycle*12-1}): {count:,} occurrences ({percentage:.1f}%)")
    
    print(f"\nDuring Regular Market Hours, Cycle {max_cycle_regular} " +
          f"(minutes {(max_cycle_regular-1)*12}-{max_cycle_regular*12-1}) " +
          f"has the highest probability at {regular_percentages[max_cycle_regular]:.1f}%")
else:
    print("No data found for Regular Market Hours")

# Execution time
elapsed_time = time.time() - start_time
print(f"\nExecution completed in {elapsed_time:.2f} seconds")