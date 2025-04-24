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
    r'c:\sqlite\nq-ohlcv-1m.csv',
    r'c:\sqlite\CME-NQ\nq-ohlcv-1m.csv'
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
output_path = os.path.join(os.path.dirname(data_path), 'nq-ohlcv-30m.csv')

print(f"Input file: {data_path}")
print(f"Output will be saved to: {output_path}")

# Load the 1-minute data
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
    
    # Check if we have enough data to resample
    if len(symbol_data) < 2:
        print(f"  Skipping {symbol} - insufficient data ({len(symbol_data)} rows)")
        continue
    
    # Sort by datetime to ensure correct resampling
    symbol_data = symbol_data.sort_values('datetime')
    
    # Set datetime as index for resampling
    symbol_data.set_index('datetime', inplace=True)
    
    # Store metadata columns
    if 'rtype' in symbol_data.columns:
        rtype = symbol_data['rtype'].iloc[0]
    else:
        rtype = 35  # Default value
    
    if 'publisher_id' in symbol_data.columns:
        publisher_id = symbol_data['publisher_id'].iloc[0]
    else:
        publisher_id = 1  # Default value
        
    if 'instrument_id' in symbol_data.columns:
        instrument_id = symbol_data['instrument_id'].iloc[0]
    else:
        instrument_id = 0  # Default value
    
    # Resample to 5-minute intervals
    try:
        resampled = symbol_data.resample('30min').agg({
            'open': 'first',      # First price in interval
            'high': 'max',        # Highest price in interval
            'low': 'min',         # Lowest price in interval
            'close': 'last',      # Last price in interval
            'volume': 'sum'       # Sum of volume in interval
        })
        
        # Drop rows with NaN values
        resampled = resampled.dropna()
        
        # Skip if resampling resulted in empty data
        if len(resampled) == 0:
            print(f"  Skipping {symbol} - no data after resampling")
            continue
            
        # Add back metadata columns
        resampled['symbol'] = symbol
        resampled['rtype'] = rtype
        resampled['publisher_id'] = publisher_id
        resampled['instrument_id'] = instrument_id
        
        # Convert index back to nanosecond timestamp
        resampled['ts_event'] = resampled.index.astype(int).astype(str)
        
        # Reset index to get datetime as a column
        resampled.reset_index(inplace=True)
        
        # Add to results
        dfs_5m.append(resampled)
        print(f"  Created {len(resampled):,} 5-minute candles for {symbol}")
        
    except Exception as e:
        print(f"  Error processing {symbol}: {e}")
    
    # Clean up to free memory
    del symbol_data

# Combine all resampled symbol data
if not dfs_5m:
    print("No data was generated! Check your input file and settings.")
    sys.exit(1)

print("Combining all symbols...")
df_5m = pd.concat(dfs_5m, ignore_index=True)

# Create final columns structure to match original format
all_columns = [
    'ts_event', 'rtype', 'publisher_id', 'instrument_id',
    'open', 'high', 'low', 'close', 'volume', 'symbol'
]

# Filter to only include columns that are available
available_columns = [col for col in all_columns if col in df_5m.columns]

# Rearrange columns to match original format
df_5m = df_5m[available_columns]

# Sort by symbol and timestamp
df_5m = df_5m.sort_values(['symbol', 'ts_event'])

# Save to CSV
print(f"Saving {len(df_5m):,} rows of 5-minute data...")
df_5m.to_csv(output_path, index=False)

# Calculate compression ratio
compression_ratio = len(df_5m) / len(df_1m) * 100
print(f"Compression ratio: {compression_ratio:.2f}% (expected ~20%)")

# Execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nExecution completed in {elapsed_time:.2f} seconds")

if elapsed_time > 60:
    print(f"That's {elapsed_time / 60:.2f} minutes")

print(f"\nSaved 5-minute data to: {output_path}")
print("Done!")