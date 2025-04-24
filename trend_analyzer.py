import pandas as pd
from datetime import time

# --- Prerequisites Block ---
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
        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True, errors='coerce')

        # 2. Drop rows where the conversion resulted in NaT in the 'ts_event' COLUMN.
        df.dropna(subset=['ts_event'], inplace=True)

        # 3. Set the cleaned 'ts_event' column as the DataFrame index.
        df.set_index('ts_event', inplace=True)

        print("Data reloaded, cleaned, and index set successfully.")
    else:
         print("DataFrame 'df' already loaded and seems valid.")

except NameError:
    print("Error: DataFrame 'df' somehow referenced before assignment.")
    exit()
except KeyError as e:
     print(f"KeyError during data check/reload: {e}. Please ensure the CSV file has the 'ts_event' column.")
     exit()
except Exception as e:
    print(f"An unexpected error occurred during data check/reload: {e}")
    exit()
# --- End Prerequisites ---


def analyze_daily_range_comparison(df_full, symbol, open_range_start_str, open_range_end_str, timezone='America/New_York'):
    """
    Analyzes and compares price direction of an opening range vs. the full day (09:30-16:00).

    Args:
        df_full (pd.DataFrame): The complete DataFrame with all symbols.
        symbol (str): The instrument symbol to filter and analyze.
        open_range_start_str (str): Opening range start time in 'HH:MM' format.
        open_range_end_str (str): Opening range end time in 'HH:MM' format.
        timezone (str): The timezone for interpreting times and grouping days.

    Returns:
        dict: A dictionary containing comparison statistics, or None if errors occur.
    """
    print(f"\n--- Analyzing Symbol: {symbol} ---")
    print(f"Opening Range: {open_range_start_str} - {open_range_end_str} ({timezone})")
    print(f"Full Day Range: 09:30 - 16:00 ({timezone})")

    try:
        # Parse user input times
        open_range_start_time = pd.to_datetime(open_range_start_str, format='%H:%M').time()
        open_range_end_time = pd.to_datetime(open_range_end_str, format='%H:%M').time()
        # Define fixed full day times
        full_day_start_time = time(9, 30)
        full_day_end_time = time(16, 0)
    except ValueError:
        print(f"Error: Invalid time format provided. Please use HH:MM.")
        return None

    # Filter for the chosen symbol
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

    # Initialize counters
    continuation_days = 0 # Opening range direction matches full day direction
    reversal_days = 0     # Opening range direction opposes full day direction
    other_days = 0        # Cases like neutral ranges or mismatch directions
    processed_days = 0    # Days where *both* ranges had valid, comparable data

    # print(f"Processing {len(unique_dates)} unique dates for {symbol}...") # Optional verbose logging

    for current_date in unique_dates:
        # --- 1. Opening Range Data ---
        open_range_start_dt = pd.Timestamp.combine(current_date, open_range_start_time).tz_localize(timezone)
        open_range_end_dt = pd.Timestamp.combine(current_date, open_range_end_time).tz_localize(timezone)

        open_range_start_data = df_symbol.asof(open_range_start_dt)
        open_range_end_data = df_symbol.asof(open_range_end_dt)

        valid_or_start = isinstance(open_range_start_data, pd.Series) and not open_range_start_data.empty and open_range_start_data.name.date() == current_date.date()
        valid_or_end = isinstance(open_range_end_data, pd.Series) and not open_range_end_data.empty and open_range_end_data.name.date() == current_date.date()
        valid_opening_range = valid_or_start and valid_or_end and open_range_end_data.name >= open_range_start_data.name # Allow start==end time

        open_range_start_open = None
        open_range_end_close = None
        if valid_opening_range:
            open_range_start_open = open_range_start_data['open']
            open_range_end_close = open_range_end_data['close']

        # --- 2. Full Day Range Data (09:30 - 16:00) ---
        full_day_start_dt = pd.Timestamp.combine(current_date, full_day_start_time).tz_localize(timezone)
        full_day_end_dt = pd.Timestamp.combine(current_date, full_day_end_time).tz_localize(timezone)

        full_day_start_data = df_symbol.asof(full_day_start_dt)
        full_day_end_data = df_symbol.asof(full_day_end_dt)

        valid_fd_start = isinstance(full_day_start_data, pd.Series) and not full_day_start_data.empty and full_day_start_data.name.date() == current_date.date()
        valid_fd_end = isinstance(full_day_end_data, pd.Series) and not full_day_end_data.empty and full_day_end_data.name.date() == current_date.date()
        valid_full_day_range = valid_fd_start and valid_fd_end and full_day_end_data.name >= full_day_start_data.name

        full_day_start_open = None
        full_day_end_close = None
        if valid_full_day_range:
             # Get the 'open' at 9:30 and 'close' at 16:00
             full_day_start_open = full_day_start_data['open']
             full_day_end_close = full_day_end_data['close']

        # --- 3. Determine Directions & Compare (Only if both ranges are valid) ---
        if valid_opening_range and valid_full_day_range:
            processed_days += 1 # Count day only if both ranges are analyzable

            # Determine opening range direction (+1 Bullish, -1 Bearish, 0 Neutral)
            opening_range_direction = 0
            if open_range_end_close > open_range_start_open: opening_range_direction = 1
            elif open_range_end_close < open_range_start_open: opening_range_direction = -1

            # Determine full day direction (+1 Bullish, -1 Bearish, 0 Neutral)
            full_day_direction = 0
            if full_day_end_close > full_day_start_open: full_day_direction = 1
            elif full_day_end_close < full_day_start_open: full_day_direction = -1

            # Compare directions (ignore neutral ranges for continuation/reversal counts)
            if opening_range_direction != 0 and full_day_direction != 0:
                if opening_range_direction == full_day_direction:
                    continuation_days += 1
                else: # Opposite directions
                    reversal_days += 1
            else: # One or both ranges were neutral, or other non-comparable outcome
                other_days += 1
        # else: # Optional: Log days skipped because one range was invalid
             # print(f"Skipping {current_date.date()}: Invalid data for one or both ranges.")


    print(f"Finished processing for {symbol}. Analyzed {processed_days} days where both ranges were valid.")

    if processed_days == 0:
        return {'symbol': symbol, 'continuation_pct': 0, 'reversal_pct': 0, 'other_pct': 0, 'processed_days': 0}

    # Calculate percentages based on days where comparison was possible
    continuation_pct = (continuation_days / processed_days) * 100
    reversal_pct = (reversal_days / processed_days) * 100
    other_pct = (other_days / processed_days) * 100

    return {
        'symbol': symbol,
        'continuation_pct': continuation_pct,
        'reversal_pct': reversal_pct,
        'other_pct': other_pct, # Includes days where one/both ranges were neutral
        'processed_days': processed_days
    }

# --- Main Execution ---

# Get opening range time input from user
input_start_time = input("Enter opening range START time (HH:MM, e.g., 09:30): ")
input_end_time = input("Enter opening range END time (HH:MM, e.g., 10:00): ")
target_timezone = 'America/New_York' # Define timezone

# Find unique symbols
unique_symbols = df['symbol'].unique()
print(f"\nFound symbols: {', '.join(unique_symbols)}")

all_results = []

# Loop through each symbol and perform the analysis
for sym in unique_symbols:
    result = analyze_daily_range_comparison(df, sym, input_start_time, input_end_time, timezone=target_timezone)
    if result:
        all_results.append(result)

# Print the collected results
print("\n--- Overall Analysis Results ---")
if not all_results:
    print("No results generated for any symbol.")
else:
    for res in all_results:
        print(f"\nSymbol: {res['symbol']}")
        print(f"  Analyzed days (both ranges valid): {res['processed_days']}")
        if res['processed_days'] > 0:
             print(f"  Continuation: {res['continuation_pct']:.2f}% (Opening range direction matched full day 09:30-16:00)")
             print(f"  Reversal:     {res['reversal_pct']:.2f}% (Opening range direction opposed full day 09:30-16:00)")
             print(f"  Other:        {res['other_pct']:.2f}% (Includes days with neutral range(s))")
        else:
             print("  No valid days found for comparison.")