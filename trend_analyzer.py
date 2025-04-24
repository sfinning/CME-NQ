import pandas as pd
from datetime import time

# --- Prerequisites ---
# Ensure the DataFrame 'df' is loaded and correctly formatted
try:
    # Check if df exists in the current scope and if its index is a DatetimeIndex
    if 'df' not in globals() or not isinstance(df.index, pd.DatetimeIndex):
        print("Reloading data...")
        csv_url = "https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv"
        df = pd.read_csv(csv_url)

        # Check if 'ts_event' column exists after loading
        if 'ts_event' not in df.columns:
            raise KeyError("The column 'ts_event' was not found in the loaded CSV file.")

        # 1. Convert 'ts_event' column to datetime objects (UTC).
        #    Invalid parsing will be set as NaT (Not a Time).
        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True, errors='coerce')

        # 2. Drop rows where the conversion resulted in NaT in the 'ts_event' COLUMN.
        #    This must be done BEFORE setting the index.
        df.dropna(subset=['ts_event'], inplace=True)

        # 3. Now, set the cleaned 'ts_event' column as the DataFrame index.
        df.set_index('ts_event', inplace=True)

        print("Data reloaded, cleaned, and index set successfully.")
    else:
         # Optional: Add a message confirming df is already loaded
         print("DataFrame 'df' already loaded and seems valid.")

# Handle potential errors during the process
except NameError:
    # This specific error shouldn't happen with 'df' not in globals() check, but good practice
    print("Error: DataFrame 'df' somehow referenced before assignment.")
    exit()
except KeyError as e:
     # Catch the specific error if 'ts_event' column is missing from CSV
     print(f"KeyError during data check/reload: {e}. Please ensure the CSV file has the 'ts_event' column.")
     exit()
except Exception as e:
    # Catch any other unexpected errors during loading/processing
    print(f"An unexpected error occurred during data check/reload: {e}")
    exit()
# --- End Prerequisites ---


def analyze_daily_price_action_per_symbol(df_full, symbol, start_time_str, end_time_str, timezone='America/New_York'):
    """
    Analyzes daily price action between a start and end time for a specific symbol.
    (This function is essentially the same as the previous version)

    Args:
        df_full (pd.DataFrame): The complete DataFrame with all symbols.
        symbol (str): The instrument symbol to filter and analyze (e.g., 'NQM0').
        start_time_str (str): Start time in 'HH:MM' format.
        end_time_str (str): End time in 'HH:MM' format.
        timezone (str): The timezone to interpret start/end times and group days.

    Returns:
        dict: A dictionary containing bullish and bearish percentages, or None if errors occur.
    """
    print(f"\n--- Analyzing Symbol: {symbol} ---")
    print(f"Time window: {start_time_str} - {end_time_str} ({timezone})")

    try:
        start_time = pd.to_datetime(start_time_str, format='%H:%M').time()
        end_time = pd.to_datetime(end_time_str, format='%H:%M').time()
    except ValueError:
        print(f"Error: Invalid time format for symbol {symbol}. Please use HH:MM.")
        return None

    # Filter for the chosen symbol *within the function*
    df_symbol = df_full[df_full['symbol'] == symbol].copy()
    if df_symbol.empty:
        print(f"No data found for symbol '{symbol}'. Skipping.")
        return None

    # Convert index to the target timezone
    try:
        df_symbol.index = df_symbol.index.tz_convert(timezone)
    except Exception as e:
        print(f"Error converting timezone for symbol {symbol}: {e}")
        return None

    # Ensure the index is sorted
    df_symbol.sort_index(inplace=True)

    unique_dates = df_symbol.index.normalize().unique()

    bullish_days = 0
    bearish_days = 0
    neutral_days = 0
    processed_days = 0

    # print(f"Processing {len(unique_dates)} unique dates for {symbol}...") # Optional: more verbose logging

    for current_date in unique_dates:
        start_dt = pd.Timestamp.combine(current_date, start_time).tz_localize(timezone)
        end_dt = pd.Timestamp.combine(current_date, end_time).tz_localize(timezone)

        start_data = df_symbol.asof(start_dt)
        end_data = df_symbol.asof(end_dt)

        valid_start = isinstance(start_data, pd.Series) and not start_data.empty and start_data.name.date() == current_date.date()
        valid_end = isinstance(end_data, pd.Series) and not end_data.empty and end_data.name.date() == current_date.date()

        if valid_start and valid_end and end_data.name > start_data.name:
            start_open = start_data['open']
            end_close = end_data['close']
            processed_days += 1

            if end_close > start_open:
                bullish_days += 1
            elif end_close < start_open:
                bearish_days += 1
            else:
                neutral_days += 1

    print(f"Finished processing for {symbol}. Analyzed {processed_days} valid days.")

    if processed_days == 0:
        return {'symbol': symbol, 'bullish_pct': 0, 'bearish_pct': 0, 'neutral_pct': 0, 'processed_days': 0}

    bullish_pct = (bullish_days / processed_days) * 100
    bearish_pct = (bearish_days / processed_days) * 100
    neutral_pct = (neutral_days / processed_days) * 100

    return {
        'symbol': symbol,
        'bullish_pct': bullish_pct,
        'bearish_pct': bearish_pct,
        'neutral_pct': neutral_pct,
        'processed_days': processed_days
    }

# --- User Input and Execution ---

# Get time input from user (once for all symbols)
input_start_time = input("Enter start time (HH:MM, e.g., 09:30): ")
input_end_time = input("Enter end time (HH:MM, e.g., 16:00): ")
target_timezone = 'America/New_York' # Define timezone

# Find unique symbols in the main DataFrame
unique_symbols = df['symbol'].unique()
print(f"\nFound symbols: {', '.join(unique_symbols)}")

all_results = []

# Loop through each symbol and perform the analysis
for sym in unique_symbols:
    # Pass the full DataFrame 'df' and the current symbol 'sym'
    result = analyze_daily_price_action_per_symbol(df, sym, input_start_time, input_end_time, timezone=target_timezone)
    if result:
        all_results.append(result)

# Print the collected results
print("\n--- Overall Analysis Results ---")
if not all_results:
    print("No results generated for any symbol.")
else:
    for res in all_results:
        print(f"\nSymbol: {res['symbol']}")
        print(f"  Processed days: {res['processed_days']}")
        if res['processed_days'] > 0:
             print(f"  Bullish Days: {res['bullish_pct']:.2f}%")
             print(f"  Bearish Days: {res['bearish_pct']:.2f}%")
             print(f"  Neutral Days: {res['neutral_pct']:.2f}%")
        else:
             print("  No valid days found in the specified time window.")