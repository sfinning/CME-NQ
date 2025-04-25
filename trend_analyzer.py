# --- Existing Prerequisites Block (Keep As Is) ---
import pandas as pd
from datetime import time
import matplotlib.pyplot as plt

# Ensure the DataFrame 'df' is loaded and correctly formatted
# (Assuming the previous loading block is here and works)
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


def analyze_daily_range_comparison(df_full, symbol,
                                   open_range_start_str, open_range_end_str,
                                   full_day_start_str, full_day_end_str, # <-- New parameters
                                   timezone='America/New_York'):
    """
    Analyzes price direction comparison, separating results by opening range direction,
    using user-defined opening and full day ranges.

    Args:
        df_full (pd.DataFrame): The full DataFrame with time-series data.
        symbol (str): The specific symbol to analyze (e.g., 'NQM4 Curncy').
        open_range_start_str (str): Start time for the opening range (HH:MM format).
        open_range_end_str (str): End time for the opening range (HH:MM format).
        full_day_start_str (str): Start time for the full day range (HH:MM format). # <-- New Doc
        full_day_end_str (str): End time for the full day range (HH:MM format).   # <-- New Doc
        timezone (str): The timezone to use for analysis (e.g., 'America/New_York').

    Returns:
        dict: A dictionary containing detailed comparison statistics (counts),
              or None if errors occur or no data is found.
    """
    print(f"\n--- Analyzing Symbol: {symbol} ---")
    try:
        # Parse user input times
        open_range_start_time = pd.to_datetime(open_range_start_str, format='%H:%M').time()
        open_range_end_time = pd.to_datetime(open_range_end_str, format='%H:%M').time()
        # Parse full day times <-- New parsing
        full_day_start_time = pd.to_datetime(full_day_start_str, format='%H:%M').time()
        full_day_end_time = pd.to_datetime(full_day_end_str, format='%H:%M').time()
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

    # Initialize detailed counters (same as before)
    bull_open_continue_bull_day = 0
    bull_open_reverse_bear_day = 0
    bull_open_neutral_day = 0
    bear_open_continue_bear_day = 0
    bear_open_reverse_bull_day = 0
    bear_open_neutral_day = 0
    neutral_open_days = 0
    valid_bullish_open_days = 0
    valid_bearish_open_days = 0
    processed_days_count = 0

    for current_date in unique_dates:
        # --- 1. Opening Range Data ---
        # (Logic remains the same, uses open_range_start_time, open_range_end_time)
        open_range_start_dt = pd.Timestamp.combine(current_date, open_range_start_time).tz_localize(timezone)
        open_range_end_dt = pd.Timestamp.combine(current_date, open_range_end_time).tz_localize(timezone)
        open_range_start_data = df_symbol.asof(open_range_start_dt)
        open_range_end_data = df_symbol.asof(open_range_end_dt)
        valid_or_start = isinstance(open_range_start_data, pd.Series) and not open_range_start_data.empty and open_range_start_data.name.date() == current_date.date()
        valid_or_end = isinstance(open_range_end_data, pd.Series) and not open_range_end_data.empty and open_range_end_data.name.date() == current_date.date()
        # Ensure end time is not before start time on the same day for asof results
        valid_opening_range = valid_or_start and valid_or_end and open_range_end_data.name >= open_range_start_data.name
        open_range_start_open = None
        open_range_end_close = None
        if valid_opening_range:
            open_range_start_open = open_range_start_data['open']
            open_range_end_close = open_range_end_data['close']


        # --- 2. Full Day Range Data ---
        # (Logic updated to use the parsed full_day_start_time, full_day_end_time) <-- Updated logic
        full_day_start_dt = pd.Timestamp.combine(current_date, full_day_start_time).tz_localize(timezone) # Use parsed time
        full_day_end_dt = pd.Timestamp.combine(current_date, full_day_end_time).tz_localize(timezone)     # Use parsed time
        full_day_start_data = df_symbol.asof(full_day_start_dt)
        full_day_end_data = df_symbol.asof(full_day_end_dt)
        valid_fd_start = isinstance(full_day_start_data, pd.Series) and not full_day_start_data.empty and full_day_start_data.name.date() == current_date.date()
        valid_fd_end = isinstance(full_day_end_data, pd.Series) and not full_day_end_data.empty and full_day_end_data.name.date() == current_date.date()
         # Ensure end time is not before start time on the same day for asof results
        valid_full_day_range = valid_fd_start and valid_fd_end and full_day_end_data.name >= full_day_start_data.name
        full_day_start_open = None
        full_day_end_close = None
        if valid_full_day_range:
             full_day_start_open = full_day_start_data['open']
             full_day_end_close = full_day_end_data['close']

        # --- 3. Determine Directions & Compare ---
        # (Logic remains the same)
        if valid_opening_range and valid_full_day_range:
            processed_days_count += 1

            opening_range_direction = 0
            if open_range_end_close > open_range_start_open: opening_range_direction = 1
            elif open_range_end_close < open_range_start_open: opening_range_direction = -1

            full_day_direction = 0
            if full_day_end_close > full_day_start_open: full_day_direction = 1
            elif full_day_end_close < full_day_start_open: full_day_direction = -1

            if opening_range_direction == 1:
                valid_bullish_open_days += 1
                if full_day_direction == 1: bull_open_continue_bull_day += 1
                elif full_day_direction == -1: bull_open_reverse_bear_day += 1
                else: bull_open_neutral_day += 1
            elif opening_range_direction == -1:
                valid_bearish_open_days += 1
                if full_day_direction == -1: bear_open_continue_bear_day += 1
                elif full_day_direction == 1: bear_open_reverse_bull_day += 1
                else: bear_open_neutral_day += 1
            else:
                neutral_open_days += 1

    print(f"Finished processing for {symbol}. Analyzed {processed_days_count} days where both ranges were valid.")

    # --- 4. Return Detailed Counts ---
    # (Return structure remains the same)
    return {
        'symbol': symbol,
        'processed_days': processed_days_count,
        'valid_bullish_open_days': valid_bullish_open_days,
        'valid_bearish_open_days': valid_bearish_open_days,
        'neutral_open_days': neutral_open_days,
        'bull_open_continue_bull_day': bull_open_continue_bull_day,
        'bull_open_reverse_bear_day': bull_open_reverse_bear_day,
        'bull_open_neutral_day': bull_open_neutral_day,
        'bear_open_continue_bear_day': bear_open_continue_bear_day,
        'bear_open_reverse_bull_day': bear_open_reverse_bull_day,
        'bear_open_neutral_day': bear_open_neutral_day,
    }

# --- Main Execution ---

# Get opening range time input from user
input_start_time = input("Enter opening range START time (HH:MM, e.g., 09:30): ")
input_end_time = input("Enter opening range END time (HH:MM, e.g., 10:00): ")

# Get full day range time input from user <-- New inputs
input_full_day_start_time = input("Enter full day START time (HH:MM, e.g., 09:30): ")
input_full_day_end_time = input("Enter full day END time (HH:MM, e.g., 16:00): ")

target_timezone = 'America/New_York' # Define timezone

# Find unique symbols
unique_symbols = df['symbol'].unique()
print(f"\nFound symbols: {', '.join(unique_symbols)}")

all_results = []

# Loop through each symbol and perform the analysis
for sym in unique_symbols:
    # Pass the new time strings to the function <-- Updated function call
    result = analyze_daily_range_comparison(
        df, sym,
        input_start_time, input_end_time,
        input_full_day_start_time, input_full_day_end_time, # Pass new times
        timezone=target_timezone
    )
    if result:
        all_results.append(result)

# --- Aggregate Detailed Results Across All Symbols ---
# (Aggregation logic remains the same)
total_processed_days = 0
total_valid_bullish_open_days = 0
total_valid_bearish_open_days = 0
total_neutral_open_days = 0
total_bull_open_continue_bull_day = 0
total_bull_open_reverse_bear_day = 0
total_bull_open_neutral_day = 0
total_bear_open_continue_bear_day = 0
total_bear_open_reverse_bull_day = 0
total_bear_open_neutral_day = 0

for res in all_results:
    total_processed_days += res['processed_days']
    total_valid_bullish_open_days += res['valid_bullish_open_days']
    total_valid_bearish_open_days += res['valid_bearish_open_days']
    total_neutral_open_days += res['neutral_open_days']
    total_bull_open_continue_bull_day += res['bull_open_continue_bull_day']
    total_bull_open_reverse_bear_day += res['bull_open_reverse_bear_day']
    total_bull_open_neutral_day += res['bull_open_neutral_day']
    total_bear_open_continue_bear_day += res['bear_open_continue_bear_day']
    total_bear_open_reverse_bull_day += res['bear_open_reverse_bull_day']
    total_bear_open_neutral_day += res['bear_open_neutral_day']

# --- Print and Plot Aggregated Summaries ---

print("\n--- Aggregated Analysis Results (All Symbols) ---")
# Display the user-defined ranges in the summary
print(f"Opening Range Analyzed: {input_start_time} - {input_end_time}")
print(f"Full Day Range Analyzed: {input_full_day_start_time} - {input_full_day_end_time}") # <-- New print
print(f"Total days where both ranges were valid: {total_processed_days}")
print(f"Total days with Neutral opening range: {total_neutral_open_days}")

# --- Summary & Pie Chart for BULLISH Opening Range Days ---
print(f"\n--- Analysis for days with BULLISH Opening Range ({input_start_time}-{input_end_time}) ---")
if total_valid_bullish_open_days == 0:
    print("No days found with a valid bullish opening range.")
else:
    bull_continue_pct = (total_bull_open_continue_bull_day / total_valid_bullish_open_days) * 100
    bull_reverse_pct = (total_bull_open_reverse_bear_day / total_valid_bullish_open_days) * 100
    bull_neutral_pct = (total_bull_open_neutral_day / total_valid_bullish_open_days) * 100

    print(f"Total days with Bullish opening range: {total_valid_bullish_open_days}")
    print(f"  Outcome during Full Day ({input_full_day_start_time}-{input_full_day_end_time}):") # <-- Updated print context
    print(f"    -> Continued Bullish: {bull_continue_pct:.2f}% ({total_bull_open_continue_bull_day} days)")
    print(f"    -> Reversed to Bearish: {bull_reverse_pct:.2f}% ({total_bull_open_reverse_bear_day} days)")
    print(f"    -> Was Neutral:         {bull_neutral_pct:.2f}% ({total_bull_open_neutral_day} days)")

    # Pie Chart for Bullish Open
    labels = 'Continued Bullish', 'Reversed to Bearish', 'Neutral Full Day'
    sizes = [total_bull_open_continue_bull_day, total_bull_open_reverse_bear_day, total_bull_open_neutral_day]
    colors = ['#66b3ff', '#ff9999', '#99ff99'] # Blue, Red, Green
    explode = (0.05, 0, 0)

    fig_bull, ax_bull = plt.subplots()
    ax_bull.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
    ax_bull.axis('equal')
    # Updated plot title <--
    plt.title(f'Full Day ({input_full_day_start_time}-{input_full_day_end_time}) Outcome Following BULLISH Opening Range ({input_start_time}-{input_end_time})\nTotal Bullish Open Days: {total_valid_bullish_open_days}')
    plt.tight_layout()
    print("\nDisplaying pie chart for Bullish Open days...")
    plt.show()


# --- Summary & Pie Chart for BEARISH Opening Range Days ---
print(f"\n--- Analysis for days with BEARISH Opening Range ({input_start_time}-{input_end_time}) ---")
if total_valid_bearish_open_days == 0:
    print("No days found with a valid bearish opening range.")
else:
    bear_continue_pct = (total_bear_open_continue_bear_day / total_valid_bearish_open_days) * 100
    bear_reverse_pct = (total_bear_open_reverse_bull_day / total_valid_bearish_open_days) * 100
    bear_neutral_pct = (total_bear_open_neutral_day / total_valid_bearish_open_days) * 100

    print(f"Total days with Bearish opening range: {total_valid_bearish_open_days}")
    print(f"  Outcome during Full Day ({input_full_day_start_time}-{input_full_day_end_time}):") # <-- Updated print context
    print(f"    -> Continued Bearish: {bear_continue_pct:.2f}% ({total_bear_open_continue_bear_day} days)")
    print(f"    -> Reversed to Bullish: {bear_reverse_pct:.2f}% ({total_bear_open_reverse_bull_day} days)")
    print(f"    -> Was Neutral:         {bear_neutral_pct:.2f}% ({total_bear_open_neutral_day} days)")

    # Pie Chart for Bearish Open
    labels = 'Continued Bearish', 'Reversed to Bullish', 'Neutral Full Day'
    sizes = [total_bear_open_continue_bear_day, total_bear_open_reverse_bull_day, total_bear_open_neutral_day]
    colors = ['#ff9999', '#66b3ff', '#99ff99'] # Red, Blue, Green
    explode = (0.05, 0, 0)

    fig_bear, ax_bear = plt.subplots()
    ax_bear.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
    ax_bear.axis('equal')
    # Updated plot title <--
    plt.title(f'Full Day ({input_full_day_start_time}-{input_full_day_end_time}) Outcome Following BEARISH Opening Range ({input_start_time}-{input_end_time})\nTotal Bearish Open Days: {total_valid_bearish_open_days}')
    plt.tight_layout()
    print("\nDisplaying pie chart for Bearish Open days...")
    plt.show()


print("\nAnalysis complete.")