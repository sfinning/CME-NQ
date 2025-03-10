import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from zoneinfo import ZoneInfo
import time

# Start timing the execution
start_time = time.time()

# Load the data files
print("Loading data files...")
df_1m = pd.read_csv('nq-ohlcv-1m.csv')
df_1h = pd.read_csv('nq-ohlcv-1h.csv')

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

# Add minute within hour
df_1m['minute'] = df_1m['datetime'].dt.minute

# Add hour start time for matching with hourly data
df_1m['hour_start'] = df_1m['datetime'].dt.floor('H')

print("Analyzing 1-minute data to identify highs and lows...")

# Define a function to analyze each hour
def analyze_hour(group):
    """Analyze a group of 1-minute data for a specific hour."""
    # Skip incomplete hours (expect at least 50 minutes)
    if len(group) < 50:
        return None
    
    # Get the hourly high and low
    hourly_high = group['high'].max()
    hourly_low = group['low'].min()
    
    # Get the first 12 minutes
    first_12_min = group[group['minute'] < 12]
    
    # If we don't have data for the first 12 minutes, skip
    if len(first_12_min) < 10:
        return None
    
    # Find the high and low in the first 12 minutes
    first_12_high = first_12_min['high'].max()
    first_12_low = first_12_min['low'].min()
    
    # Check if first 12 minutes contain the high or low
    high_in_first_12 = abs(first_12_high - hourly_high) < 0.01
    low_in_first_12 = abs(first_12_low - hourly_low) < 0.01
    
    # Exact match check as backup
    if not high_in_first_12:
        high_in_first_12 = hourly_high in first_12_min['high'].values
    if not low_in_first_12:
        low_in_first_12 = hourly_low in first_12_min['low'].values
    
    return {
        'hour': group['hour'].iloc[0],
        'day_of_week': group['day_of_week'].iloc[0],
        'symbol': group['symbol'].iloc[0],
        'date': group['date'].iloc[0],
        'high_in_first_12': high_in_first_12,
        'low_in_first_12': low_in_first_12,
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

# Calculate probabilities by hour of day
print("Calculating probabilities by hour of day...")
hourly_stats = results_df.groupby('hour').agg({
    'high_in_first_12': 'mean',
    'low_in_first_12': 'mean',
    'either_in_first_12': 'mean',
    'both_in_first_12': 'mean',
    'symbol': 'count'
}).reset_index()

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

hourly_stats['market_context'] = hourly_stats['hour'].apply(get_chicago_market_context)

# Print results
print("\nProbability of First 12 Minutes Setting Hourly High/Low (Chicago Time)")
print("=" * 90)
print("{:<6} {:<15} {:<15} {:<15} {:<15} {:<10} {:<20}".format(
    "Hour", "High Prob %", "Low Prob %", "Either %", "Both %", "Samples", "Market Context (CT)"))
print("-" * 90)

for _, row in hourly_stats.iterrows():
    print("{:<6} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<10} {:<20}".format(
        f"{int(row['hour']):02d}:00", 
        row['high_in_first_12'] * 100,
        row['low_in_first_12'] * 100,
        row['either_in_first_12'] * 100,
        row['both_in_first_12'] * 100,
        row['symbol'],
        row['market_context']
    ))

# Calculate overall probabilities
overall_high_prob = results_df['high_in_first_12'].mean() * 100
overall_low_prob = results_df['low_in_first_12'].mean() * 100
overall_either_prob = results_df['either_in_first_12'].mean() * 100
overall_both_prob = results_df['both_in_first_12'].mean() * 100

print("\nOverall Probabilities:")
print(f"High set in first 12 minutes: {overall_high_prob:.2f}%")
print(f"Low set in first 12 minutes: {overall_low_prob:.2f}%")
print(f"Either high or low set in first 12 minutes: {overall_either_prob:.2f}%")
print(f"Both high and low set in first 12 minutes: {overall_both_prob:.2f}%")

# Create visualization
plt.figure(figsize=(14, 10))

# Plot 1: Probabilities by hour
ax1 = plt.subplot(2, 1, 1)
width = 0.25
x = np.arange(len(hourly_stats))
ax1.bar(x - width, hourly_stats['high_in_first_12']*100, width, label='High in First 12m', color='#2a9d8f')
ax1.bar(x, hourly_stats['low_in_first_12']*100, width, label='Low in First 12m', color='#e76f51')
ax1.bar(x + width, hourly_stats['either_in_first_12']*100, width, label='Either High or Low', color='#457b9d')

# Add market hour context - highlight CME regular hours
ax1.axvspan(8, 15.5, alpha=0.2, color='green', label='CME Regular Hours')
ax1.axvline(x=hourly_stats[hourly_stats['hour'] == 8].index[0], color='red', linestyle='--', alpha=0.7, label='CME Open (8:30 CT)')
ax1.axvline(x=hourly_stats[hourly_stats['hour'] == 15].index[0], color='red', linestyle='--', alpha=0.7, label='CME Close (15:15 CT)')

ax1.set_title('Probability of First 12 Minutes Setting Hourly High/Low (Chicago Time)', fontsize=14)
ax1.set_ylabel('Probability (%)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{h:02d}:00" for h in hourly_stats['hour']])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add reference lines for overall probabilities
ax1.axhline(y=overall_high_prob, color='#2a9d8f', linestyle='--', alpha=0.5)
ax1.axhline(y=overall_low_prob, color='#e76f51', linestyle='--', alpha=0.5)
ax1.axhline(y=overall_either_prob, color='#457b9d', linestyle='--', alpha=0.5)

# Plot 2: Sample sizes
ax2 = plt.subplot(2, 1, 2)
ax2.bar(hourly_stats['hour'], hourly_stats['symbol'], color='#5a189a', alpha=0.8)
ax2.set_title('Sample Size by Hour (Chicago Time)', fontsize=14)
ax2.set_xlabel('Hour of Day (24h format CT)', fontsize=12)
ax2.set_ylabel('Number of Observations', fontsize=12)
ax2.set_xticks(range(0, 24))
ax2.grid(axis='y', alpha=0.3)
ax2.axvspan(8, 15.5, alpha=0.2, color='green')

plt.tight_layout()
plt.savefig('first_12min_probability_chicago.png')

# Group by market session for CME trading
cme_sessions = {
    'CME Pre-Market (6:00-8:30)': hourly_stats[(hourly_stats['hour'] >= 6) & (hourly_stats['hour'] < 9)],
    'CME Open (8:30-10:00)': hourly_stats[(hourly_stats['hour'] == 8) | (hourly_stats['hour'] == 9)],
    'CME Morning (10:00-12:00)': hourly_stats[(hourly_stats['hour'] >= 10) & (hourly_stats['hour'] < 12)],
    'CME Midday (12:00-14:00)': hourly_stats[(hourly_stats['hour'] >= 12) & (hourly_stats['hour'] < 14)],
    'CME Afternoon (14:00-15:15)': hourly_stats[(hourly_stats['hour'] == 14) | (hourly_stats['hour'] == 15)],
    'CME Post-Close (15:15-17:00)': hourly_stats[(hourly_stats['hour'] == 15) | (hourly_stats['hour'] == 16)],
    'CME Overnight (17:00-6:00)': hourly_stats[(hourly_stats['hour'] > 16) | (hourly_stats['hour'] < 6)]
}

print("\nProbabilities by CME Market Session (Chicago Time)")
print("=" * 90)
print("{:<25} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Session", "High %", "Low %", "Either %", "Both %", "Samples"))
print("-" * 90)

for session, data in cme_sessions.items():
    if not data.empty:
        high_prob = data['high_in_first_12'].mean() * 100
        low_prob = data['low_in_first_12'].mean() * 100
        either_prob = data['either_in_first_12'].mean() * 100
        both_prob = data['both_in_first_12'].mean() * 100
        samples = data['symbol'].sum()
        
        print("{:<25} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
            session, high_prob, low_prob, either_prob, both_prob, samples
        ))

# Day of week analysis
day_of_week_stats = results_df.groupby('day_of_week').agg({
    'high_in_first_12': 'mean',
    'low_in_first_12': 'mean',
    'either_in_first_12': 'mean',
    'both_in_first_12': 'mean',
    'symbol': 'count'
}).reset_index()

day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday'}
day_of_week_stats['day_name'] = day_of_week_stats['day_of_week'].map(day_names)

print("\nProbabilities by Day of Week")
print("=" * 75)
print("{:<12} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Day", "High %", "Low %", "Either %", "Both %", "Samples"))
print("-" * 75)

for _, row in day_of_week_stats.iterrows():
    print("{:<12} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
        row['day_name'],
        row['high_in_first_12'] * 100,
        row['low_in_first_12'] * 100,
        row['either_in_first_12'] * 100,
        row['both_in_first_12'] * 100,
        row['symbol']
    ))

# Specific market-opening hours (8am and 9am)
key_hours = [8, 9]
key_hours_stats = hourly_stats[hourly_stats['hour'].isin(key_hours)]

print("\nDetailed Analysis for CME Market Opening Hours")
print("=" * 75)
print("{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<20}".format(
    "Hour", "High %", "Low %", "Either %", "Both %", "Samples", "Context"))
print("-" * 75)

for _, row in key_hours_stats.iterrows():
    print("{:<6} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10} {:<20}".format(
        f"{int(row['hour']):02d}:00",
        row['high_in_first_12'] * 100,
        row['low_in_first_12'] * 100,
        row['either_in_first_12'] * 100,
        row['both_in_first_12'] * 100,
        row['symbol'],
        row['market_context']
    ))

# Execution time
end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.2f} seconds")