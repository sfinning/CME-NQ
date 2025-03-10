import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from zoneinfo import ZoneInfo
import time
import sys

# Start timing the execution
start_time = time.time()

# Define file paths
# Use correct paths to access Git LFS files
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)  # Go up one level from CME-NQ folder

# Try multiple possible locations for the data files
possible_paths = [
    # Current directory
    os.path.join(base_dir, 'nq-ohlcv-1m.csv'),
    os.path.join(base_dir, 'nq-ohlcv-1h.csv'),
    
    # Parent directory
    os.path.join(parent_dir, 'nq-ohlcv-1m.csv'),
    os.path.join(parent_dir, 'nq-ohlcv-1h.csv'),
    
    # Absolute paths
    r'c:\sqlite\nq-ohlcv-1m.csv',
    r'c:\sqlite\nq-ohlcv-1h.csv'
]

# Find the first available path for each file
path_1m = None
path_1h = None

for path in possible_paths:
    if os.path.exists(path):
        if '1m.csv' in path and path_1m is None:
            path_1m = path
        elif '1h.csv' in path and path_1h is None:
            path_1h = path
    
    if path_1m is not None and path_1h is not None:
        break

# Check if files were found
if path_1m is None or path_1h is None:
    print("Error: Could not find data files. Please ensure they exist in one of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

# Print the paths being used
print(f"Using 1-minute data file: {path_1m}")
print(f"Using 1-hour data file: {path_1h}")

# Load the data files
print("Loading data files...")
df_1m = pd.read_csv(path_1m)
df_1h = pd.read_csv(path_1h)

# Convert timestamps from nanoseconds to datetime (UTC)
print("Processing timestamps...")
df_1m['datetime_utc'] = pd.to_datetime(df_1m['ts_event'], unit='ns')
df_1h['datetime_utc'] = pd.to_datetime(df_1h['ts_event'], unit='ns')

# Convert UTC to Chicago time (Central Time)
chicago_tz = ZoneInfo('America/Chicago')
df_1m['datetime'] = df_1m['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert(chicago_tz).dt.tz_localize(None)
df_1h['datetime'] = df_1h['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert(chicago_tz).dt.tz_localize(None)

# Add hour of day column (Chicago time)
df_1m['hour'] = df_1m['datetime'].dt.hour
df_1h['hour'] = df_1h['datetime'].dt.hour

# Add day of week (1=Monday, 7=Sunday)
df_1m['day_of_week'] = df_1m['datetime'].dt.dayofweek + 1
df_1h['day_of_week'] = df_1h['datetime'].dt.dayofweek + 1

# Add a date column for grouping
df_1m['date'] = df_1m['datetime'].dt.date
df_1h['date'] = df_1h['datetime'].dt.date

# Filter to standard contracts (exclude spread symbols) with meaningful volume
df_1m = df_1m[~df_1m['symbol'].str.contains('-')]
df_1m = df_1m[df_1m['volume'] > 0]
df_1h = df_1h[~df_1h['symbol'].str.contains('-')]
df_1h = df_1h[df_1h['volume'] > 0]

# Filter out weekends (Saturday = 6, Sunday = 7)
df_1m = df_1m[df_1m['day_of_week'] < 6]
df_1h = df_1h[df_1h['day_of_week'] < 6]

# Classify hourly candles as bullish or bearish
df_1h['candle_direction'] = np.where(df_1h['close'] > df_1h['open'], 'bullish', 'bearish')

# Add minute within hour
df_1m['minute'] = df_1m['datetime'].dt.minute

# Add hour start time for matching with hourly data
# Fix: Change 'H' to 'h' to address the FutureWarning
df_1m['hour_start'] = df_1m['datetime'].dt.floor('h')

# Create a mapping from hourly data to identify bullish/bearish candles
# Create unique identifier for each hour
df_1h['hour_id'] = df_1h['symbol'] + '_' + df_1h['datetime'].dt.strftime('%Y-%m-%d_%H')
df_1h_direction = df_1h[['hour_id', 'candle_direction']]

# Create the same identifier in minute data for merging
df_1m['hour_id'] = df_1m['symbol'] + '_' + df_1m['hour_start'].dt.strftime('%Y-%m-%d_%H')

# Map the candle direction to each minute data point
df_1m = pd.merge(df_1m, df_1h_direction, on='hour_id', how='left')

print("Analyzing 1-minute data to identify highs and lows...")

# Define a function to analyze each hour with extended time periods
def analyze_hour(group):
    """Analyze a group of 1-minute data for a specific hour."""
    # Skip incomplete hours (expect at least 50 minutes)
    if len(group) < 50:
        return None
    
    # Get the hourly high and low
    hourly_high = group['high'].max()
    hourly_low = group['low'].min()
    
    # Get the first 12, 24, and 36 minutes
    first_12_min = group[group['minute'] < 12]
    first_24_min = group[group['minute'] < 24]
    first_36_min = group[group['minute'] < 36]
    
    # If we don't have data for the first periods, skip
    if len(first_12_min) < 10 or len(first_24_min) < 20 or len(first_36_min) < 30:
        return None
    
    # Get candle direction (should be the same for all rows in the group)
    candle_direction = group['candle_direction'].iloc[0]
    if pd.isna(candle_direction):
        return None
    
    # Find the high and low in each time period
    first_12_high = first_12_min['high'].max()
    first_12_low = first_12_min['low'].min()
    first_24_high = first_24_min['high'].max()
    first_24_low = first_24_min['low'].min()
    first_36_high = first_36_min['high'].max()
    first_36_low = first_36_min['low'].min()
    
    # Check if first periods contain the high or low (with small tolerance for floating point comparison)
    high_in_first_12 = abs(first_12_high - hourly_high) < 0.01
    low_in_first_12 = abs(first_12_low - hourly_low) < 0.01
    high_in_first_24 = abs(first_24_high - hourly_high) < 0.01
    low_in_first_24 = abs(first_24_low - hourly_low) < 0.01
    high_in_first_36 = abs(first_36_high - hourly_high) < 0.01
    low_in_first_36 = abs(first_36_low - hourly_low) < 0.01
    
    # Exact match check as backup
    if not high_in_first_12 and hourly_high in first_12_min['high'].values:
        high_in_first_12 = True
    if not low_in_first_12 and hourly_low in first_12_min['low'].values:
        low_in_first_12 = True
    if not high_in_first_24 and hourly_high in first_24_min['high'].values:
        high_in_first_24 = True
    if not low_in_first_24 and hourly_low in first_24_min['low'].values:
        low_in_first_24 = True
    if not high_in_first_36 and hourly_high in first_36_min['high'].values:
        high_in_first_36 = True
    if not low_in_first_36 and hourly_low in first_36_min['low'].values:
        low_in_first_36 = True
    
    return {
        'hour': group['hour'].iloc[0],
        'day_of_week': group['day_of_week'].iloc[0],
        'symbol': group['symbol'].iloc[0],
        'date': group['date'].iloc[0],
        'candle_direction': candle_direction,
        'high_in_first_12': high_in_first_12,
        'low_in_first_12': low_in_first_12,
        'high_in_first_24': high_in_first_24,
        'low_in_first_24': low_in_first_24,
        'high_in_first_36': high_in_first_36,
        'low_in_first_36': low_in_first_36,
        'either_in_first_12': high_in_first_12 or low_in_first_12,
        'both_in_first_12': high_in_first_12 and low_in_first_12
    }

print("Grouping and analyzing hourly patterns...")
# Group by symbol, date, and hour_start
grouped = df_1m.groupby(['symbol', 'date', 'hour_start'])

# Apply the analysis function to each group
results = []
for name, group in grouped:
    result = analyze_hour(group)
    if result:
        results.append(result)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Calculate probabilities by hour of day and candle direction
print("Calculating probabilities by hour of day and candle direction...")

# Split results into bullish and bearish
results_bullish = results_df[results_df['candle_direction'] == 'bullish']
results_bearish = results_df[results_df['candle_direction'] == 'bearish']

# Calculate stats for each candle direction by hour
hourly_stats_bullish = results_bullish.groupby('hour').agg({
    'high_in_first_12': 'mean',
    'low_in_first_12': 'mean',
    'low_in_first_24': 'mean',
    'low_in_first_36': 'mean',
    'either_in_first_12': 'mean',
    'both_in_first_12': 'mean',
    'symbol': 'count'
}).reset_index()

hourly_stats_bearish = results_bearish.groupby('hour').agg({
    'high_in_first_12': 'mean',
    'high_in_first_24': 'mean',
    'high_in_first_36': 'mean',
    'low_in_first_12': 'mean',
    'either_in_first_12': 'mean',
    'both_in_first_12': 'mean',
    'symbol': 'count'
}).reset_index()

# Calculate overall probabilities regardless of hour
overall_bullish = {
    'high_12': results_bullish['high_in_first_12'].mean() * 100,
    'low_12': results_bullish['low_in_first_12'].mean() * 100,
    'low_24': results_bullish['low_in_first_24'].mean() * 100,
    'low_36': results_bullish['low_in_first_36'].mean() * 100,
    'either': results_bullish['either_in_first_12'].mean() * 100,
    'both': results_bullish['both_in_first_12'].mean() * 100,
    'samples': len(results_bullish)
}

overall_bearish = {
    'high_12': results_bearish['high_in_first_12'].mean() * 100,
    'high_24': results_bearish['high_in_first_24'].mean() * 100,
    'high_36': results_bearish['high_in_first_36'].mean() * 100,
    'low_12': results_bearish['low_in_first_12'].mean() * 100,
    'either': results_bearish['either_in_first_12'].mean() * 100,
    'both': results_bearish['both_in_first_12'].mean() * 100,
    'samples': len(results_bearish)
}

# Add CME/Chicago market context
def get_chicago_market_context(hour):
    if hour == 8:
        return "CME Open (8:30 CT)"
    elif hour == 15:
        return "CME Close (15:15 CT)"
    elif 8 <= hour <= 15:
        return "Regular Trading Hours"
    elif 16 <= hour <= 17:
        return "CME Globex Close/Reopen"
    elif hour < 8 or hour >= 18:
        return "CME Globex Session"
    else:
        return "Post-Close"

hourly_stats_bullish['market_context'] = hourly_stats_bullish['hour'].apply(get_chicago_market_context)
hourly_stats_bearish['market_context'] = hourly_stats_bearish['hour'].apply(get_chicago_market_context)

# Print extended results for bullish candles (focus on LOW)
print("\n\nEXTENDED ANALYSIS BY CANDLE DIRECTION (CHICAGO TIME)")
print("=" * 100)

print("\nH1 BULLISH CANDLES - LOW PROBABILITY BY TIME PERIOD")
print("=" * 90)
print("{:<6} {:<15} {:<15} {:<15} {:<10} {:<20}".format(
    "Hour", "Low in 12m %", "Low in 24m %", "Low in 36m %", "Samples", "Market Context (CT)"))
print("-" * 90)

for _, row in hourly_stats_bullish.iterrows():
    print("{:<6} {:<15.2f} {:<15.2f} {:<15.2f} {:<10} {:<20}".format(
        f"{int(row['hour']):02d}:00", 
        row['low_in_first_12'] * 100,
        row['low_in_first_24'] * 100,
        row['low_in_first_36'] * 100,
        row['symbol'],
        row['market_context']
    ))

# Print extended results for bearish candles (focus on HIGH)
print("\nH1 BEARISH CANDLES - HIGH PROBABILITY BY TIME PERIOD")
print("=" * 90)
print("{:<6} {:<15} {:<15} {:<15} {:<10} {:<20}".format(
    "Hour", "High in 12m %", "High in 24m %", "High in 36m %", "Samples", "Market Context (CT)"))
print("-" * 90)

for _, row in hourly_stats_bearish.iterrows():
    print("{:<6} {:<15.2f} {:<15.2f} {:<15.2f} {:<10} {:<20}".format(
        f"{int(row['hour']):02d}:00", 
        row['high_in_first_12'] * 100,
        row['high_in_first_24'] * 100,
        row['high_in_first_36'] * 100,
        row['symbol'],
        row['market_context']
    ))

# Print overall probabilities with extended time periods
print("\nOVERALL PROBABILITIES FOR KEY PATTERNS:")
print("-" * 80)
print("BULLISH CANDLES - Probability of Low being set in first N minutes:")
print(f"  First 12 minutes: {overall_bullish['low_12']:.2f}%")
print(f"  First 24 minutes: {overall_bullish['low_24']:.2f}%")
print(f"  First 36 minutes: {overall_bullish['low_36']:.2f}%")
print(f"  Sample size: {overall_bullish['samples']} candles\n")

print("BEARISH CANDLES - Probability of High being set in first N minutes:")
print(f"  First 12 minutes: {overall_bearish['high_12']:.2f}%")
print(f"  First 24 minutes: {overall_bearish['high_24']:.2f}%")
print(f"  First 36 minutes: {overall_bearish['high_36']:.2f}%")
print(f"  Sample size: {overall_bearish['samples']} candles")

# First show the standard cumulative probability analysis
print("\nOVERALL PROBABILITIES FOR KEY PATTERNS (CUMULATIVE):")
print("-" * 80)
print("BULLISH CANDLES - Probability of Low being set by minute N:")
print(f"  First 12 minutes: {overall_bullish['low_12']:.2f}%")
print(f"  First 24 minutes: {overall_bullish['low_24']:.2f}%")
print(f"  First 36 minutes: {overall_bullish['low_36']:.2f}%")
print(f"  Sample size: {overall_bullish['samples']} candles\n")

print("BEARISH CANDLES - Probability of High being set by minute N:")
print(f"  First 12 minutes: {overall_bearish['high_12']:.2f}%")
print(f"  First 24 minutes: {overall_bearish['high_24']:.2f}%")
print(f"  First 36 minutes: {overall_bearish['high_36']:.2f}%")
print(f"  Sample size: {overall_bearish['samples']} candles")

# Calculate and show discrete time window probabilities
# For bullish candles - when the low is set
hourly_stats_bullish['low_0_12m'] = hourly_stats_bullish['low_in_first_12']
hourly_stats_bullish['low_12_24m'] = hourly_stats_bullish['low_in_first_24'] - hourly_stats_bullish['low_in_first_12']
hourly_stats_bullish['low_24_36m'] = hourly_stats_bullish['low_in_first_36'] - hourly_stats_bullish['low_in_first_24']
hourly_stats_bullish['low_36_60m'] = 1.0 - hourly_stats_bullish['low_in_first_36']

# For bearish candles - when the high is set
hourly_stats_bearish['high_0_12m'] = hourly_stats_bearish['high_in_first_12']
hourly_stats_bearish['high_12_24m'] = hourly_stats_bearish['high_in_first_24'] - hourly_stats_bearish['high_in_first_12']
hourly_stats_bearish['high_24_36m'] = hourly_stats_bearish['high_in_first_36'] - hourly_stats_bearish['high_in_first_24']
hourly_stats_bearish['high_36_60m'] = 1.0 - hourly_stats_bearish['high_in_first_36']

# Calculate overall averages for discrete time windows
avg_bullish_windows = {
    'low_0_12': overall_bullish['low_12'],
    'low_12_24': overall_bullish['low_24'] - overall_bullish['low_12'],
    'low_24_36': overall_bullish['low_36'] - overall_bullish['low_24'],
    'low_36_60': 100 - overall_bullish['low_36']
}

avg_bearish_windows = {
    'high_0_12': overall_bearish['high_12'],
    'high_12_24': overall_bearish['high_24'] - overall_bearish['high_12'],
    'high_24_36': overall_bearish['high_36'] - overall_bearish['high_24'],
    'high_36_60': 100 - overall_bearish['high_36']
}

# Print overall statistics for discrete time windows
print("\nOVERALL TIMING DISTRIBUTION BY TIME WINDOW:")
print("=" * 60)
print("BULLISH CANDLES - When is the Low typically set:")
print(f"  Minutes 0-12:  {avg_bullish_windows['low_0_12']:.2f}%")
print(f"  Minutes 12-24: {avg_bullish_windows['low_12_24']:.2f}%")
print(f"  Minutes 24-36: {avg_bullish_windows['low_24_36']:.2f}%")
print(f"  Minutes 36-60: {avg_bullish_windows['low_36_60']:.2f}%\n")

print("BEARISH CANDLES - When is the High typically set:")
print(f"  Minutes 0-12:  {avg_bearish_windows['high_0_12']:.2f}%")
print(f"  Minutes 12-24: {avg_bearish_windows['high_12_24']:.2f}%")
print(f"  Minutes 24-36: {avg_bearish_windows['high_24_36']:.2f}%")
print(f"  Minutes 36-60: {avg_bearish_windows['high_36_60']:.2f}%")
print(f"  Sample size: {overall_bearish['samples']} candles")

# Print detailed breakdown by hour in discrete time windows
print("\n\nDETAILED TIME WINDOW ANALYSIS BY HOUR (CHICAGO TIME)")
print("=" * 100)

# Create enhanced visualizations for time progression
plt.figure(figsize=(16, 10))

# BULLISH LOW PROGRESSION CHART
ax1 = plt.subplot(2, 1, 1)
width = 0.2
x = np.arange(len(hourly_stats_bullish))
ax1.bar(x - width, hourly_stats_bullish['low_in_first_12']*100, width, label='Low in First 12m', color='#2a9d8f')
ax1.bar(x, hourly_stats_bullish['low_in_first_24']*100, width, label='Low in First 24m', color='#457b9d')
ax1.bar(x + width, hourly_stats_bullish['low_in_first_36']*100, width, label='Low in First 36m', color='#e76f51')

# Add market hour context
ax1.axvspan(8, 15.5, alpha=0.2, color='green', label='CME Regular Hours')
ax1.set_title('H1 BULLISH Candles: Probability of Low Being Set in First N Minutes', fontsize=14)
ax1.set_ylabel('Probability (%)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{h:02d}:00" for h in hourly_stats_bullish['hour']])
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add reference lines for overall probabilities
ax1.axhline(y=overall_bullish['low_12'], color='#2a9d8f', linestyle='--', alpha=0.5)
ax1.axhline(y=overall_bullish['low_24'], color='#457b9d', linestyle='--', alpha=0.5)
ax1.axhline(y=overall_bullish['low_36'], color='#e76f51', linestyle='--', alpha=0.5)

# BEARISH HIGH PROGRESSION CHART
ax2 = plt.subplot(2, 1, 2)
x = np.arange(len(hourly_stats_bearish))
ax2.bar(x - width, hourly_stats_bearish['high_in_first_12']*100, width, label='High in First 12m', color='#2a9d8f')
ax2.bar(x, hourly_stats_bearish['high_in_first_24']*100, width, label='High in First 24m', color='#457b9d')
ax2.bar(x + width, hourly_stats_bearish['high_in_first_36']*100, width, label='High in First 36m', color='#e76f51')

# Add market hour context
ax2.axvspan(8, 15.5, alpha=0.2, color='green', label='CME Regular Hours')
ax2.set_title('H1 BEARISH Candles: Probability of High Being Set in First N Minutes', fontsize=14)
ax2.set_ylabel('Probability (%)', fontsize=12)
ax2.set_xlabel('Hour of Day (Chicago Time)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{h:02d}:00" for h in hourly_stats_bearish['hour']])
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Add reference lines for overall probabilities
ax2.axhline(y=overall_bearish['high_12'], color='#2a9d8f', linestyle='--', alpha=0.5)
ax2.axhline(y=overall_bearish['high_24'], color='#457b9d', linestyle='--', alpha=0.5)
ax2.axhline(y=overall_bearish['high_36'], color='#e76f51', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('first_12_24_36_analysis.png')

# Compute direct comparisons between bullish and bearish
# Merge the datasets on hour
combined_stats = pd.merge(
    hourly_stats_bullish[['hour', 'high_in_first_12', 'low_in_first_12', 'symbol']], 
    hourly_stats_bearish[['hour', 'high_in_first_12', 'low_in_first_12', 'symbol']], 
    on='hour', 
    suffixes=('_bull', '_bear')
)

# Calculate differences (bullish minus bearish)
combined_stats['high_diff'] = (combined_stats['high_in_first_12_bull'] - combined_stats['high_in_first_12_bear'])*100
combined_stats['low_diff'] = (combined_stats['low_in_first_12_bull'] - combined_stats['low_in_first_12_bear'])*100

# Print comparison table
print("\nDIFFERENCE IN PROBABILITIES (Bullish - Bearish)")
print("=" * 70)
print("{:<6} {:<15} {:<15} {:<15} {:<15}".format(
    "Hour", "High Diff %", "Low Diff %", "Bull Samples", "Bear Samples"))
print("-" * 70)

for _, row in combined_stats.iterrows():
    print("{:<6} {:<+15.2f} {:<+15.2f} {:<15} {:<15}".format(
        f"{int(row['hour']):02d}:00", 
        row['high_diff'],
        row['low_diff'],
        row['symbol_bull'],
        row['symbol_bear']
    ))

# Calculate probabilities for discrete time windows
# For bullish candles - when the low is set
hourly_stats_bullish['low_0_12m'] = hourly_stats_bullish['low_in_first_12']
hourly_stats_bullish['low_12_24m'] = hourly_stats_bullish['low_in_first_24'] - hourly_stats_bullish['low_in_first_12']
hourly_stats_bullish['low_24_36m'] = hourly_stats_bullish['low_in_first_36'] - hourly_stats_bullish['low_in_first_24']
hourly_stats_bullish['low_36_60m'] = 1.0 - hourly_stats_bullish['low_in_first_36']

# For bearish candles - when the high is set
hourly_stats_bearish['high_0_12m'] = hourly_stats_bearish['high_in_first_12']
hourly_stats_bearish['high_12_24m'] = hourly_stats_bearish['high_in_first_24'] - hourly_stats_bearish['high_in_first_12']
hourly_stats_bearish['high_24_36m'] = hourly_stats_bearish['high_in_first_36'] - hourly_stats_bearish['high_in_first_24']
hourly_stats_bearish['high_36_60m'] = 1.0 - hourly_stats_bearish['high_in_first_36']

# Print results in discrete time windows
print("\n\nDISCRETE TIME WINDOW ANALYSIS (CHICAGO TIME)")
print("=" * 100)

print("\nH1 BULLISH CANDLES - WHEN THE LOW IS SET (BY TIME WINDOW)")
print("=" * 100)
print("{:<6} {:<12} {:<12} {:<12} {:<12} {:<10} {:<20}".format(
    "Hour", "0-12m %", "12-24m %", "24-36m %", "36-60m %", "Samples", "Market Context (CT)"))
print("-" * 100)

for _, row in hourly_stats_bullish.iterrows():
    print("{:<6} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<10} {:<20}".format(
        f"{int(row['hour']):02d}:00", 
        row['low_0_12m'] * 100,
        row['low_12_24m'] * 100,
        row['low_24_36m'] * 100,
        row['low_36_60m'] * 100,
        row['symbol'],
        row['market_context']
    ))

print("\nH1 BEARISH CANDLES - WHEN THE HIGH IS SET (BY TIME WINDOW)")
print("=" * 100)
print("{:<6} {:<12} {:<12} {:<12} {:<12} {:<10} {:<20}".format(
    "Hour", "0-12m %", "12-24m %", "24-36m %", "36-60m %", "Samples", "Market Context (CT)"))
print("-" * 100)

for _, row in hourly_stats_bearish.iterrows():
    print("{:<6} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<10} {:<20}".format(
        f"{int(row['hour']):02d}:00", 
        row['high_0_12m'] * 100,
        row['high_12_24m'] * 100,
        row['high_24_36m'] * 100,
        row['high_36_60m'] * 100,
        row['symbol'],
        row['market_context']
    ))

# Create visualizations for discrete time windows
plt.figure(figsize=(16, 10))

# BULLISH LOW DISCRETE TIME WINDOWS
ax1 = plt.subplot(2, 1, 1)
width = 0.2
x = np.arange(len(hourly_stats_bullish))

# Create stacked bar chart for discrete time windows
ax1.bar(x, hourly_stats_bullish['low_0_12m']*100, width*2, 
        label='Low in 0-12m', color='#2a9d8f')
ax1.bar(x, hourly_stats_bullish['low_12_24m']*100, width*2, 
        bottom=hourly_stats_bullish['low_0_12m']*100, 
        label='Low in 12-24m', color='#457b9d')
ax1.bar(x, hourly_stats_bullish['low_24_36m']*100, width*2, 
        bottom=(hourly_stats_bullish['low_0_12m'] + hourly_stats_bullish['low_12_24m'])*100, 
        label='Low in 24-36m', color='#e76f51')
ax1.bar(x, hourly_stats_bullish['low_36_60m']*100, width*2, 
        bottom=(hourly_stats_bullish['low_0_12m'] + hourly_stats_bullish['low_12_24m'] + hourly_stats_bullish['low_24_36m'])*100, 
        label='Low in 36-60m', color='#9d4edd')

# Add market hour context
ax1.axvspan(8, 15.5, alpha=0.2, color='green', label='CME Regular Hours')
ax1.set_title('H1 BULLISH Candles: Distribution of When the Low is Set (Chicago Time)', fontsize=14)
ax1.set_ylabel('Probability (%)', fontsize=12)
ax1.set_yticks(range(0, 101, 10))
ax1.set_xticks(x)
ax1.set_xticklabels([f"{h:02d}:00" for h in hourly_stats_bullish['hour']])
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# BEARISH HIGH DISCRETE TIME WINDOWS
ax2 = plt.subplot(2, 1, 2)
x = np.arange(len(hourly_stats_bearish))

# Create stacked bar chart for discrete time windows
ax2.bar(x, hourly_stats_bearish['high_0_12m']*100, width*2, 
        label='High in 0-12m', color='#2a9d8f')
ax2.bar(x, hourly_stats_bearish['high_12_24m']*100, width*2, 
        bottom=hourly_stats_bearish['high_0_12m']*100, 
        label='High in 12-24m', color='#457b9d')
ax2.bar(x, hourly_stats_bearish['high_24_36m']*100, width*2, 
        bottom=(hourly_stats_bearish['high_0_12m'] + hourly_stats_bearish['high_12_24m'])*100, 
        label='High in 24-36m', color='#e76f51')
ax2.bar(x, hourly_stats_bearish['high_36_60m']*100, width*2, 
        bottom=(hourly_stats_bearish['high_0_12m'] + hourly_stats_bearish['high_12_24m'] + hourly_stats_bearish['high_24_36m'])*100, 
        label='High in 36-60m', color='#9d4edd')

# Add market hour context
ax2.axvspan(8, 15.5, alpha=0.2, color='green', label='CME Regular Hours')
ax2.set_title('H1 BEARISH Candles: Distribution of When the High is Set (Chicago Time)', fontsize=14)
ax2.set_ylabel('Probability (%)', fontsize=12)
ax2.set_yticks(range(0, 101, 10))
ax2.set_xlabel('Hour of Day (Chicago Time)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{h:02d}:00" for h in hourly_stats_bearish['hour']])
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('time_window_distribution.png')

# Calculate overall averages for discrete time windows
avg_bullish_windows = {
    'low_0_12': overall_bullish['low_12'],
    'low_12_24': overall_bullish['low_24'] - overall_bullish['low_12'],
    'low_24_36': overall_bullish['low_36'] - overall_bullish['low_24'],
    'low_36_60': 100 - overall_bullish['low_36']
}

avg_bearish_windows = {
    'high_0_12': overall_bearish['high_12'],
    'high_12_24': overall_bearish['high_24'] - overall_bearish['high_12'],
    'high_24_36': overall_bearish['high_36'] - overall_bearish['high_24'],
    'high_36_60': 100 - overall_bearish['high_36']
}

# Print overall statistics for discrete time windows
print("\nOVERALL TIMING DISTRIBUTION BY CANDLE DIRECTION:")
print("=" * 60)
print("BULLISH CANDLES - When is the Low typically set:")
print(f"  Minutes 0-12:  {avg_bullish_windows['low_0_12']:.2f}%")
print(f"  Minutes 12-24: {avg_bullish_windows['low_12_24']:.2f}%")
print(f"  Minutes 24-36: {avg_bullish_windows['low_24_36']:.2f}%")
print(f"  Minutes 36-60: {avg_bullish_windows['low_36_60']:.2f}%")
print(f"  Sample size: {overall_bullish['samples']} candles\n")

print("BEARISH CANDLES - When is the High typically set:")
print(f"  Minutes 0-12:  {avg_bearish_windows['high_0_12']:.2f}%")
print(f"  Minutes 12-24: {avg_bearish_windows['high_12_24']:.2f}%")
print(f"  Minutes 24-36: {avg_bearish_windows['high_24_36']:.2f}%")
print(f"  Minutes 36-60: {avg_bearish_windows['high_36_60']:.2f}%")
print(f"  Sample size: {overall_bearish['samples']} candles")

# Create a pie chart visualization of the overall timing distributions
plt.figure(figsize=(14, 7))

# BULLISH LOW PIE CHART
ax1 = plt.subplot(1, 2, 1)
labels = ['0-12 minutes', '12-24 minutes', '24-36 minutes', '36-60 minutes']
sizes = [
    avg_bullish_windows['low_0_12'], 
    avg_bullish_windows['low_12_24'], 
    avg_bullish_windows['low_24_36'], 
    avg_bullish_windows['low_36_60']
]
colors = ['#2a9d8f', '#457b9d', '#e76f51', '#9d4edd']
explode = (0.1, 0, 0, 0)  # explode the 1st slice (0-12 minutes)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
ax1.set_title('When Lows are Set in BULLISH H1 Candles', fontsize=14)

# BEARISH HIGH PIE CHART
ax2 = plt.subplot(1, 2, 2)
sizes = [
    avg_bearish_windows['high_0_12'], 
    avg_bearish_windows['high_12_24'], 
    avg_bearish_windows['high_24_36'], 
    avg_bearish_windows['high_36_60']
]

ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.axis('equal')
ax2.set_title('When Highs are Set in BEARISH H1 Candles', fontsize=14)

plt.tight_layout()
plt.savefig('time_window_distribution_pie.png')

# Execution time
end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")