import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
import time

# Start timing execution
start_time = time.time()

print("=" * 80)
print("ANALYSIS: PROBABILITY OF TRADING THROUGH PREVIOUS HOUR'S HIGH")
print("=" * 80)
print("This script analyzes: When first 5m high equals 1h high,")
print("what's the probability of trading through that high in the next hour?")
print("-" * 80)

# Find the necessary data files
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Define potential file paths
data_paths = {
    "5m": [
        os.path.join(base_dir, 'nq-ohlcv-5m.csv'),
        os.path.join(parent_dir, 'nq-ohlcv-5m.csv'),
        r'c:\sqlite\nq-ohlcv-5m.csv'
    ],
    "1h": [
        os.path.join(base_dir, 'nq-ohlcv-1h.csv'),
        os.path.join(parent_dir, 'nq-ohlcv-1h.csv'),
        r'c:\sqlite\nq-ohlcv-1h.csv'
    ]
}

# Find the available files
file_paths = {}
for timeframe, paths in data_paths.items():
    for path in paths:
        if os.path.exists(path):
            file_paths[timeframe] = path
            break

# Check if we have both files needed
if "5m" not in file_paths or "1h" not in file_paths:
    print("Error: Could not find both 5m and 1h data files.")
    if "5m" not in file_paths:
        print("Missing: 5-minute data file")
    if "1h" not in file_paths:
        print("Missing: 1-hour data file")
    sys.exit(1)

print(f"Using 5-minute data: {file_paths['5m']}")
print(f"Using 1-hour data: {file_paths['1h']}")

# Load the data files
print("\nLoading 5-minute data...")
df_5m = pd.read_csv(file_paths['5m'])
print(f"Loaded {len(df_5m):,} rows of 5-minute data")

print("Loading 1-hour data...")
df_1h = pd.read_csv(file_paths['1h'])
print(f"Loaded {len(df_1h):,} rows of 1-hour data")

# Convert timestamps to datetime
print("\nProcessing timestamps...")
df_5m['datetime'] = pd.to_datetime(df_5m['ts_event'], unit='ns')
df_1h['datetime'] = pd.to_datetime(df_1h['ts_event'], unit='ns')

# Extract hour start times for both datasets
df_5m['hour_start'] = df_5m['datetime'].dt.floor('h')
df_5m['next_hour_start'] = df_5m['hour_start'] + timedelta(hours=1)

df_1h['hour_start'] = df_1h['datetime'].dt.floor('h')
df_1h['next_hour_start'] = df_1h['hour_start'] + timedelta(hours=1)

# Extract date, hour, minute for easier grouping
df_5m['date'] = df_5m['datetime'].dt.date
df_5m['hour'] = df_5m['datetime'].dt.hour
df_5m['minute'] = df_5m['datetime'].dt.minute
df_5m['day_of_week'] = df_5m['datetime'].dt.dayofweek + 1  # 1=Monday, 7=Sunday

df_1h['date'] = df_1h['datetime'].dt.date
df_1h['hour'] = df_1h['datetime'].dt.hour
df_1h['day_of_week'] = df_1h['datetime'].dt.dayofweek + 1

# Filter out weekends
df_5m = df_5m[df_5m['day_of_week'] < 6]
df_1h = df_1h[df_1h['day_of_week'] < 6]

# Filter out spread symbols
print("Filtering out spread symbols...")
df_5m = df_5m[~df_5m['symbol'].str.contains('-')]
df_1h = df_1h[~df_1h['symbol'].str.contains('-')]

# Create unique identifiers for each hour
df_5m['hour_id'] = df_5m['symbol'] + '_' + df_5m['hour_start'].dt.strftime('%Y-%m-%d_%H')
df_5m['next_hour_id'] = df_5m['symbol'] + '_' + df_5m['next_hour_start'].dt.strftime('%Y-%m-%d_%H')

df_1h['hour_id'] = df_1h['symbol'] + '_' + df_1h['hour_start'].dt.strftime('%Y-%m-%d_%H')
df_1h['next_hour_id'] = df_1h['symbol'] + '_' + df_1h['next_hour_start'].dt.strftime('%Y-%m-%d_%H')

# Filter 5m data to only keep first bar of each hour (0-5 minutes)
df_5m_first = df_5m[df_5m['minute'] < 5].copy()

print("\nAnalyzing price patterns...")
results = []

# Process each symbol separately
symbols = sorted(set(df_5m['symbol']).intersection(set(df_1h['symbol'])))
print(f"Processing {len(symbols)} symbols...")

for symbol in symbols:
    print(f"Analyzing {symbol}...")
    
    # Get data for this symbol
    symbol_5m = df_5m[df_5m['symbol'] == symbol].copy()
    symbol_1h = df_1h[df_1h['symbol'] == symbol].copy()
    
    # Get first 5m candle of each hour
    first_5m = df_5m_first[df_5m_first['symbol'] == symbol].copy()
    
    # Create lookup dictionaries for hourly data
    hour_data = {}
    for _, row in symbol_1h.iterrows():
        hour_data[row['hour_id']] = {
            'high': row['high'],
            'low': row['low'],
            'open': row['open'],
            'close': row['close']
        }
    
    # Check each hour
    for _, first_bar in first_5m.iterrows():
        hour_id = first_bar['hour_id']
        next_hour_id = first_bar['next_hour_id']
        
        # Skip if we don't have hourly data for this hour or next hour
        if hour_id not in hour_data or next_hour_id not in hour_data:
            continue
        
        # Get the high of the hourly bar
        hour_high = hour_data[hour_id]['high']
        first_5m_high = first_bar['high']
        
        # Check if first 5m high is equal to hourly high (with small tolerance)
        is_equal = abs(first_5m_high - hour_high) < 0.01
        
        if not is_equal:
            continue
        
        # Get next hour's 5-minute data
        next_hour_5m = symbol_5m[symbol_5m['hour_id'] == next_hour_id].copy()
        
        # Skip if we don't have enough data for next hour
        if len(next_hour_5m) < 10:  # Need at least 10 bars (50 minutes of data)
            continue
            
        # Get next hour's high
        next_hour_high = next_hour_5m['high'].max()
        
        # Check if it trades above the previous hour's high
        trades_above = next_hour_high > hour_high
        
        # Calculate penetration if it trades above
        penetration = next_hour_high - hour_high if trades_above else 0
        
        # Determine how many bars into the next hour before it trades through
        bars_until_breakout = None
        if trades_above:
            # Find the first 5m bar in the next hour that trades above the high
            for i, row in next_hour_5m.sort_values('datetime').iterrows():
                if row['high'] > hour_high:
                    bars_until_breakout = i - next_hour_5m.index[0] + 1
                    break
        
        # Get directional info
        hour_direction = 1 if hour_data[hour_id]['close'] > hour_data[hour_id]['open'] else -1
        
        # Add results
        results.append({
            'symbol': symbol,
            'hour_start': first_bar['hour_start'],
            'next_hour_start': first_bar['next_hour_start'],
            'hour': first_bar['hour'],
            'hour_high': hour_high,
            'trades_above': trades_above,
            'penetration': penetration,
            'bars_until_breakout': bars_until_breakout,
            'hour_direction': hour_direction
        })

# Convert results to DataFrame
print(f"\nCompiling results from {len(results)} matching patterns...")
results_df = pd.DataFrame(results)

# Check if we have any valid results
if len(results_df) == 0:
    print("No valid instances found where first 5m high equals 1h high.")
    sys.exit(0)

# Calculate overall probability
overall_prob = results_df['trades_above'].mean() * 100
avg_penetration = results_df[results_df['trades_above']]['penetration'].mean()
median_bars_until_breakout = results_df[results_df['trades_above']]['bars_until_breakout'].median()

print("\nOVERALL RESULTS")
print("=" * 80)
print(f"Total instances where first 5m high equals 1h high: {len(results_df):,}")
print(f"Probability of trading above this high in next hour: {overall_prob:.2f}%")
print(f"When broken, average penetration above the high: {avg_penetration:.2f} points")
print(f"Median 5m bars until breakout: {median_bars_until_breakout:.1f} bars")

# Calculate by hour of day
by_hour = results_df.groupby('hour').agg({
    'trades_above': ['mean', 'count'],
    'penetration': ['mean', 'median']
})

by_hour.columns = ['probability', 'samples', 'avg_penetration', 'med_penetration']
by_hour['probability'] *= 100
by_hour = by_hour.reset_index()

# Add market context
def get_market_context(hour):
    if hour == 8:
        return "CME Open (8:30 CT)"
    elif hour == 15:
        return "CME Close (15:15 CT)"
    elif 8 <= hour <= 15:
        return "Regular Trading"
    else:
        return "Extended Hours"

by_hour['context'] = by_hour['hour'].apply(get_market_context)

# Print results by hour
print("\nRESULTS BY HOUR OF DAY")
print("=" * 100)
print("{:<6} {:<15} {:<15} {:<15} {:<20}".format(
    "Hour", "Probability %", "Avg Penetration", "Samples", "Market Context"))
print("-" * 100)

for _, row in by_hour.iterrows():
    print("{:<6} {:<15.2f} {:<15.2f} {:<15} {:<20}".format(
        f"{int(row['hour']):02d}:00", 
        row['probability'],
        row['avg_penetration'],
        row['samples'],
        row['context']
    ))

# Compare market sessions
regular_hours = by_hour[(by_hour['hour'] >= 8) & (by_hour['hour'] <= 15)]
extended_hours = by_hour[(by_hour['hour'] < 8) | (by_hour['hour'] > 15)]

reg_prob = regular_hours['probability'].mean() if len(regular_hours) > 0 else 0
ext_prob = extended_hours['probability'].mean() if len(extended_hours) > 0 else 0

reg_samples = regular_hours['samples'].sum() if len(regular_hours) > 0 else 0
ext_samples = extended_hours['samples'].sum() if len(extended_hours) > 0 else 0

print("\nSESSION COMPARISON")
print("=" * 80)
print(f"Regular Trading Hours (8:00-15:00): {reg_prob:.2f}% ({reg_samples:,} samples)")
print(f"Extended Trading Hours: {ext_prob:.2f}% ({ext_samples:,} samples)")

# Analyze by hour direction (bullish vs bearish)
bullish_hours = results_df[results_df['hour_direction'] > 0]
bearish_hours = results_df[results_df['hour_direction'] < 0]

bull_prob = bullish_hours['trades_above'].mean() * 100 if len(bullish_hours) > 0 else 0
bear_prob = bearish_hours['trades_above'].mean() * 100 if len(bearish_hours) > 0 else 0

print("\nHOUR DIRECTION ANALYSIS")
print("=" * 80)
print(f"After bullish hours: {bull_prob:.2f}% ({len(bullish_hours):,} samples)")
print(f"After bearish hours: {bear_prob:.2f}% ({len(bearish_hours):,} samples)")

# Penetration depth analysis
penetration_stats = results_df[results_df['trades_above']]['penetration'].describe()

print("\nPENETRATION STATISTICS (WHEN HIGH IS BROKEN)")
print("=" * 80)
print(f"Count: {penetration_stats['count']:.0f}")
print(f"Mean: {penetration_stats['mean']:.2f} points")
print(f"Median: {penetration_stats['50%']:.2f} points")
print(f"Standard Deviation: {penetration_stats['std']:.2f} points")
print(f"Minimum: {penetration_stats['min']:.2f} points")
print(f"Maximum: {penetration_stats['max']:.2f} points")
print(f"25th percentile: {penetration_stats['25%']:.2f} points")
print(f"75th percentile: {penetration_stats['75%']:.2f} points")

# Analyze time to breakout using bars_until_breakout
breakout_timing = results_df[results_df['trades_above']]['bars_until_breakout'].describe()

print("\nTIME TO BREAKOUT STATISTICS (IN 5M BARS)")
print("=" * 80)
print(f"Count: {breakout_timing['count']:.0f}")
print(f"Mean: {breakout_timing['mean']:.2f} bars")
print(f"Median: {breakout_timing['50%']:.2f} bars")
print(f"Standard Deviation: {breakout_timing['std']:.2f} bars")
print(f"Minimum: {breakout_timing['min']:.2f} bars")
print(f"Maximum: {breakout_timing['max']:.2f} bars")
print(f"25th percentile: {breakout_timing['25%']:.2f} bars")
print(f"75th percentile: {breakout_timing['75%']:.2f} bars")

# Distribution of breakout timing
bars_counts = results_df[results_df['trades_above']]['bars_until_breakout'].value_counts().sort_index()
total_breakouts = bars_counts.sum()

print("\nBREAKOUT TIMING DISTRIBUTION")
print("=" * 80)
print(f"{'Bar #':10} {'Count':10} {'Percentage':10} {'Cumulative %':15}")
print("-" * 80)

cumulative = 0
for bars, count in bars_counts.items():
    cumulative += count
    percentage = (count / total_breakouts) * 100
    cumulative_pct = (cumulative / total_breakouts) * 100
    print(f"{int(bars):10} {count:10} {percentage:.2f}%{' '*5} {cumulative_pct:.2f}%")

# Execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nExecution completed in {elapsed_time:.2f} seconds")