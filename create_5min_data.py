import pandas as pd
import os
import sys
import time

# Start timing execution
start_time = time.time()

print("=" * 60)
print("Creating 5-Minute OHLCV Data from 1-Minute Data (UTC Time)")
print("=" * 60)

# Find the 1-minute data file
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

# Define the output path based on where the input file was found
output_path = os.path.join(os.path.dirname(data_path), 'nq-ohlcv-5m.csv')

print(f"Input file: {data_path}")
print(f"Output will be saved to: {output_path}")

# Load the 1-minute data with only necessary columns
print("Loading 1-minute data...")
df_1m = pd.read_csv(data_path)

print(f"Loaded {len(df_1m):,} rows of 1-minute data")

# Convert timestamp to datetime (keeping UTC)
print("Processing timestamps...")
df_1m['datetime'] = pd.to_datetime(df_1m['ts_event'], unit='ns')

# Process each symbol separately to ensure correct resampling
symbols = df_1m['symbol'].unique()
print(f"Found {len(symbols)} unique symbols")

# Create an empty list to store resampled dataframes
dfs_5m = []

# Process each symbol
for i, symbol in enumerate(symbols):
    print(f"Processing symbol {i+1}/{len(symbols)}: {symbol}")
    
    # Filter data for this symbol
    symbol_data = df_1m[df_1m['symbol'] == symbol].copy()
    
    # Sort by datetime to ensure correct resampling
    symbol_data = symbol_data.sort_values('datetime')
    
    # Set datetime as index for resampling
    symbol_data.set_index('datetime', inplace=True)
    
    # Resample to 5-minute intervals using 'min' instead of deprecated 'T'
    resampled = symbol_data.resample('5min').agg({
        'ts_event': 'first',  # Take timestamp of first minute
        'symbol': 'first',    # Symbol remains the same
        'open': 'first',      # First price in interval
        'high': 'max',        # Highest price in interval
        'low': 'min',         # Lowest price in interval
        'close': 'last',      # Last price in interval
        'volume': 'sum'       # Sum of volume in interval
    })
    
    # Drop rows with NaN values (incomplete intervals)
    resampled = resampled.dropna()
    
    # Reset index to get datetime as a column
    resampled.reset_index(inplace=True)
    
    # Append to list of resampled dataframes
    if len(resampled) > 0:
        dfs_5m.append(resampled)
    
    # Clean up to free memory
    del symbol_data
    del resampled

# Combine all resampled symbol data
print("Combining all symbols...")
df_5m = pd.concat(dfs_5m, ignore_index=True)

# Ensure all columns match the original format
print("Formatting data...")
df_5m = df_5m[['ts_event', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

# Sort by symbol and timestamp
df_5m = df_5m.sort_values(['symbol', 'ts_event'])

# Save to CSV
print(f"Saving {len(df_5m):,} rows of 5-minute data...")
df_5m.to_csv(output_path, index=False)

# Calculate compression ratio
compression_ratio = len(df_5m) / len(df_1m) * 100 if len(df_1m) > 0 else 0
print(f"Compression ratio: {compression_ratio:.2f}% (expected ~20%)")

# Execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nExecution completed in {elapsed_time:.2f} seconds")

if elapsed_time > 60:
    print(f"That's {elapsed_time / 60:.2f} minutes")

print(f"\nSaved 5-minute data to: {output_path}")
print("Done!")