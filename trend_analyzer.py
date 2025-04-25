# --- Prerequisites Block ---
import pandas as pd
from datetime import time
import matplotlib.pyplot as plt
import sys # Import sys to allow exiting

# --- Data Loading and Preparation ---
# Ensure the DataFrame 'df' is loaded and correctly formatted
df = None  # Initialize df to avoid referencing it before assignment

try:
    # Check if df exists in the current scope and if its index is a DatetimeIndex
    # If not, load and prepare it.
    if 'df' not in globals() or not isinstance(df.index, pd.DatetimeIndex):
        print("Loading data...")
        # Define the URL for the CSV file
        csv_url = "https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv"
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_url)

        # Check if the crucial 'ts_event' column exists after loading
        if 'ts_event' not in df.columns:
            raise KeyError("The column 'ts_event' was not found in the loaded CSV file.")

        # 1. Convert 'ts_event' column (assumed to be nanoseconds since epoch) to datetime objects (UTC).
        #    'coerce' will turn unparseable values into NaT (Not a Time).
        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True, errors='coerce')

        # 2. Drop rows where the 'ts_event' conversion resulted in NaT.
        #    This cleans the data by removing entries with invalid timestamps.
        df.dropna(subset=['ts_event'], inplace=True)

        # 3. Set the cleaned 'ts_event' column as the DataFrame index.
        #    This is essential for time-series operations like 'asof'.
        df.set_index('ts_event', inplace=True)

        print("Data loaded, cleaned, and index set successfully.")
    else:
         # If 'df' already exists and seems valid, skip reloading.
         print("DataFrame 'df' already loaded and seems valid.")

# --- Error Handling for Loading ---
except NameError:
    # This error shouldn't typically happen with the check above but is a safeguard.
    print("Error: DataFrame 'df' was somehow referenced before assignment during loading.")
    sys.exit(1) # Exit the script if data loading fails critically
except KeyError as e:
     # Specific error if the required column is missing.
     print(f"KeyError during data loading: {e}. Please ensure the CSV file is correct and has the 'ts_event' column.")
     sys.exit(1)
except Exception as e:
    # Catch any other unexpected errors during the loading process.
    print(f"An unexpected error occurred during data loading/preparation: {e}")
    sys.exit(1)
# --- End Data Loading ---


# --- Core Analysis Function ---
def analyze_daily_range_comparison(df_full, symbol,
                                   obs_window_start_str, obs_window_end_str,
                                   trade_window_start_str, trade_window_end_str,
                                   timezone='America/New_York'):
    """
    Analyzes price direction comparison between an Observation Window and a Trading Window,
    separating results by the direction of the Observation Window.

    Args:
        df_full (pd.DataFrame): The full DataFrame with time-series data, indexed by UTC timestamp.
        symbol (str): The specific symbol to analyze (e.g., 'NQM4 Curncy').
        obs_window_start_str (str): Start time for the observation window (HH:MM format).
        obs_window_end_str (str): End time for the observation window (HH:MM format).
        trade_window_start_str (str): Start time for the trading window (HH:MM format).
        trade_window_end_str (str): End time for the trading window (HH:MM format).
        timezone (str): The target timezone for interpreting start/end times and analysis
                      (e.g., 'America/New_York').

    Returns:
        dict: A dictionary containing detailed comparison statistics (counts) for the symbol,
              or None if errors occur or no data is found for the symbol.
    """
    print(f"\n--- Analyzing Symbol: {symbol} ---")
    try:
        # Parse user input time strings into time objects.
        obs_window_start_time = pd.to_datetime(obs_window_start_str, format='%H:%M').time()
        obs_window_end_time = pd.to_datetime(obs_window_end_str, format='%H:%M').time()
        trade_window_start_time = pd.to_datetime(trade_window_start_str, format='%H:%M').time()
        trade_window_end_time = pd.to_datetime(trade_window_end_str, format='%H:%M').time()
    except ValueError:
        # Handle cases where the user provides time in an incorrect format.
        print(f"Error: Invalid time format provided for symbol {symbol}. Please use HH:MM.")
        return None # Return None to indicate failure for this symbol

    # Filter the main DataFrame to get data only for the specified symbol.
    # Use .copy() to avoid SettingWithCopyWarning later.
    df_symbol = df_full[df_full['symbol'] == symbol].copy()
    if df_symbol.empty:
        # If no data exists for this symbol, print a message and skip it.
        print(f"No data found for symbol '{symbol}'. Skipping.")
        return None

    # Convert the DataFrame's index (which is UTC) to the target timezone.
    # This allows comparing times using local market hours (e.g., New York time).
    try:
        df_symbol.index = df_symbol.index.tz_convert(timezone)
    except Exception as e:
        # Handle potential errors during timezone conversion.
        print(f"Error converting timezone for symbol {symbol}: {e}")
        return None

    # Ensure the DataFrame index is sorted chronologically.
    # This is crucial for time-based lookups like 'asof'.
    df_symbol.sort_index(inplace=True)

    # Get unique dates present in the data for this symbol (based on the target timezone).
    # normalize() sets the time part to 00:00:00, giving unique calendar dates.
    unique_dates = df_symbol.index.normalize().unique()

    # --- Initialize Counters ---
    # These counters track the occurrences of different outcome scenarios.
    bull_obs_continue_bull_trade = 0 # Bullish observation -> Bullish trading window
    bull_obs_reverse_bear_trade = 0  # Bullish observation -> Bearish trading window
    bull_obs_neutral_trade = 0       # Bullish observation -> Neutral trading window

    bear_obs_continue_bear_trade = 0 # Bearish observation -> Bearish trading window
    bear_obs_reverse_bull_trade = 0  # Bearish observation -> Bullish trading window
    bear_obs_neutral_trade = 0       # Bearish observation -> Neutral trading window

    neutral_obs_days = 0             # Count of days where the observation window was neutral

    valid_bullish_obs_days = 0       # Count of days with a valid Bullish observation window
    valid_bearish_obs_days = 0       # Count of days with a valid Bearish observation window
    processed_days_count = 0         # Count of days where *both* windows had valid data

    # --- Iterate Through Each Day ---
    for current_date in unique_dates:
        # --- 1. Get Observation Window Data ---
        # Combine the current date with the specified start/end times and localize to the target timezone.
        obs_window_start_dt = pd.Timestamp.combine(current_date, obs_window_start_time).tz_localize(timezone)
        obs_window_end_dt = pd.Timestamp.combine(current_date, obs_window_end_time).tz_localize(timezone)

        # Use 'asof' to find the *last known* data point at or before the specified start/end datetimes.
        # This handles cases where there isn't data at the exact microsecond.
        obs_window_start_data = df_symbol.asof(obs_window_start_dt)
        obs_window_end_data = df_symbol.asof(obs_window_end_dt)

        # --- Validate Observation Window Data ---
        # Check if 'asof' returned valid Series objects (not None or empty).
        valid_obs_start = isinstance(obs_window_start_data, pd.Series) and not obs_window_start_data.empty
        valid_obs_end = isinstance(obs_window_end_data, pd.Series) and not obs_window_end_data.empty
        # Further check: Ensure the data found belongs to the *current date* we are processing.
        # This prevents 'asof' from grabbing data from the previous day if data is missing at the start of the current day.
        if valid_obs_start: valid_obs_start = obs_window_start_data.name.date() == current_date.date()
        if valid_obs_end: valid_obs_end = obs_window_end_data.name.date() == current_date.date()
        # Ensure the end data point is not chronologically before the start data point.
        valid_obs_window = valid_obs_start and valid_obs_end and obs_window_end_data.name >= obs_window_start_data.name

        obs_window_start_open = None
        obs_window_end_close = None
        if valid_obs_window:
            # If valid, extract the 'open' price at the start and 'close' price at the end.
            obs_window_start_open = obs_window_start_data['open']
            obs_window_end_close = obs_window_end_data['close']


        # --- 2. Get Trading Window Data --- (Similar logic as above)
        trade_window_start_dt = pd.Timestamp.combine(current_date, trade_window_start_time).tz_localize(timezone)
        trade_window_end_dt = pd.Timestamp.combine(current_date, trade_window_end_time).tz_localize(timezone)
        trade_window_start_data = df_symbol.asof(trade_window_start_dt)
        trade_window_end_data = df_symbol.asof(trade_window_end_dt)

        # --- Validate Trading Window Data --- (Similar logic as above)
        valid_trade_start = isinstance(trade_window_start_data, pd.Series) and not trade_window_start_data.empty
        valid_trade_end = isinstance(trade_window_end_data, pd.Series) and not trade_window_end_data.empty
        if valid_trade_start: valid_trade_start = trade_window_start_data.name.date() == current_date.date()
        if valid_trade_end: valid_trade_end = trade_window_end_data.name.date() == current_date.date()
        valid_trade_window = valid_trade_start and valid_trade_end and trade_window_end_data.name >= trade_window_start_data.name

        trade_window_start_open = None
        trade_window_end_close = None
        if valid_trade_window:
             trade_window_start_open = trade_window_start_data['open']
             trade_window_end_close = trade_window_end_data['close']

        # --- 3. Determine Directions & Compare ---
        # Proceed only if we have valid data for *both* the observation and trading windows for the current day.
        if valid_obs_window and valid_trade_window:
            processed_days_count += 1 # Increment the count of successfully processed days

            # Determine observation window direction (1: Bullish, -1: Bearish, 0: Neutral)
            obs_window_direction = 0
            # Add a small tolerance for floating point comparison if needed, e.g., use math.isclose
            if obs_window_end_close > obs_window_start_open: obs_window_direction = 1
            elif obs_window_end_close < obs_window_start_open: obs_window_direction = -1

            # Determine trading window direction (1: Bullish, -1: Bearish, 0: Neutral)
            trade_window_direction = 0
            if trade_window_end_close > trade_window_start_open: trade_window_direction = 1
            elif trade_window_end_close < trade_window_start_open: trade_window_direction = -1

            # --- Categorize and Count ---
            # Increment the appropriate counters based on the directions found.
            if obs_window_direction == 1: # Bullish Observation Window
                valid_bullish_obs_days += 1
                if trade_window_direction == 1: bull_obs_continue_bull_trade += 1
                elif trade_window_direction == -1: bull_obs_reverse_bear_trade += 1
                else: bull_obs_neutral_trade += 1 # Trade window was neutral
            elif obs_window_direction == -1: # Bearish Observation Window
                valid_bearish_obs_days += 1
                if trade_window_direction == -1: bear_obs_continue_bear_trade += 1
                elif trade_window_direction == 1: bear_obs_reverse_bull_trade += 1
                else: bear_obs_neutral_trade += 1 # Trade window was neutral
            else: # Neutral Observation Window
                neutral_obs_days += 1
                # Note: We currently don't separately track the outcome of the trading window
                # when the observation window is neutral, but we could add counters here if needed.

    # --- Log Completion for Symbol ---
    print(f"Finished processing for {symbol}. Analyzed {processed_days_count} days where both windows were valid.")

    # --- 4. Return Results for Symbol ---
    # Package the counts into a dictionary for this specific symbol.
    return {
        'symbol': symbol,
        'processed_days': processed_days_count,
        'valid_bullish_obs_days': valid_bullish_obs_days,
        'valid_bearish_obs_days': valid_bearish_obs_days,
        'neutral_obs_days': neutral_obs_days,
        # Counts for Bullish Observation scenarios
        'bull_obs_continue_bull_trade': bull_obs_continue_bull_trade,
        'bull_obs_reverse_bear_trade': bull_obs_reverse_bear_trade,
        'bull_obs_neutral_trade': bull_obs_neutral_trade,
        # Counts for Bearish Observation scenarios
        'bear_obs_continue_bear_trade': bear_obs_continue_bear_trade,
        'bear_obs_reverse_bull_trade': bear_obs_reverse_bull_trade,
        'bear_obs_neutral_trade': bear_obs_neutral_trade,
    }
# --- End Analysis Function ---


# --- Main Execution Block ---
if __name__ == "__main__": # Ensures this block runs only when the script is executed directly

    # --- User Input for Time Windows ---
    print("Please define the time windows for the analysis.")
    # Get Observation window start and end times from the user.
    input_obs_start_time = input("Enter Observation Window START time (HH:MM, e.g., 09:30): ")
    input_obs_end_time = input("Enter Observation Window END time (HH:MM, e.g., 10:00): ")

    # Get Trading window start and end times from the user.
    input_trade_start_time = input("Enter Trading Window START time (HH:MM, e.g., 09:30): ")
    input_trade_end_time = input("Enter Trading Window END time (HH:MM, e.g., 16:00): ")

    # --- Configuration ---
    target_timezone = 'America/New_York' # Define the primary timezone for analysis

    # --- Symbol Discovery ---
    # Find all unique symbols present in the 'symbol' column of the DataFrame.
    try:
        unique_symbols = df['symbol'].unique()
        if len(unique_symbols) == 0:
             print("Error: No symbols found in the 'symbol' column of the loaded data.")
             sys.exit(1)
        print(f"\nFound symbols: {', '.join(unique_symbols)}")
    except KeyError:
        print("Error: The 'symbol' column is missing from the loaded data.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while identifying unique symbols: {e}")
        sys.exit(1)


    # --- Run Analysis for Each Symbol ---
    all_results = [] # Initialize a list to store the result dictionaries from each symbol

    # Loop through each unique symbol found in the data.
    for sym in unique_symbols:
        # Call the analysis function for the current symbol, passing the user-defined times.
        result = analyze_daily_range_comparison(
            df, sym,
            input_obs_start_time, input_obs_end_time,
            input_trade_start_time, input_trade_end_time,
            timezone=target_timezone
        )
        # If the analysis function returned a valid result (not None), add it to the list.
        if result:
            all_results.append(result)

    # Check if any results were generated
    if not all_results:
        print("\nNo valid analysis results were generated for any symbol. This might be due to:")
        print("- Data missing for the specified time windows.")
        print("- Invalid time formats entered.")
        print("- Issues with the input data file.")
        sys.exit(0) # Exit gracefully if no results

    # --- Aggregate Results Across All Symbols ---
    # Initialize aggregate counters.
    total_processed_days = 0
    total_valid_bullish_obs_days = 0
    total_valid_bearish_obs_days = 0
    total_neutral_obs_days = 0
    total_bull_obs_continue_bull_trade = 0
    total_bull_obs_reverse_bear_trade = 0
    total_bull_obs_neutral_trade = 0
    total_bear_obs_continue_bear_trade = 0
    total_bear_obs_reverse_bull_trade = 0
    total_bear_obs_neutral_trade = 0

    # Sum up the counts from each symbol's result dictionary.
    for res in all_results:
        total_processed_days += res['processed_days']
        total_valid_bullish_obs_days += res['valid_bullish_obs_days']
        total_valid_bearish_obs_days += res['valid_bearish_obs_days']
        total_neutral_obs_days += res['neutral_obs_days']
        total_bull_obs_continue_bull_trade += res['bull_obs_continue_bull_trade']
        total_bull_obs_reverse_bear_trade += res['bull_obs_reverse_bear_trade']
        total_bull_obs_neutral_trade += res['bull_obs_neutral_trade']
        total_bear_obs_continue_bear_trade += res['bear_obs_continue_bear_trade']
        total_bear_obs_reverse_bull_trade += res['bear_obs_reverse_bull_trade']
        total_bear_obs_neutral_trade += res['bear_obs_neutral_trade']

    # --- Print and Plot Aggregated Summaries ---

    print("\n--- Aggregated Analysis Results (All Symbols Combined) ---")
    print(f"Observation Window Analyzed: {input_obs_start_time} - {input_obs_end_time}")
    print(f"Trading Window Analyzed:    {input_trade_start_time} - {input_trade_end_time} ({target_timezone})")
    print(f"Total days where both windows were valid: {total_processed_days}")
    print(f"Total days with Neutral observation window: {total_neutral_obs_days}")

    # --- Summary & Pie Chart for BULLISH Observation Window Days ---
    print(f"\n--- Analysis for days starting with a BULLISH Observation Window ({input_obs_start_time}-{input_obs_end_time}) ---")
    # Check if there were any days with a valid bullish observation window to avoid division by zero.
    if total_valid_bullish_obs_days == 0:
        print("No days found with a valid bullish observation window.")
    else:
        # Calculate percentages for each outcome following a bullish observation.
        bull_continue_pct = (total_bull_obs_continue_bull_trade / total_valid_bullish_obs_days) * 100
        bull_reverse_pct = (total_bull_obs_reverse_bear_trade / total_valid_bullish_obs_days) * 100
        bull_neutral_pct = (total_bull_obs_neutral_trade / total_valid_bullish_obs_days) * 100

        # Print the summary statistics.
        print(f"Total days with Bullish observation window: {total_valid_bullish_obs_days}")
        print(f"  Outcome during Trading Window ({input_trade_start_time}-{input_trade_end_time}):")
        print(f"    -> Continued Bullish: {bull_continue_pct:.2f}% ({total_bull_obs_continue_bull_trade} days)")
        print(f"    -> Reversed to Bearish: {bull_reverse_pct:.2f}% ({total_bull_obs_reverse_bear_trade} days)")
        print(f"    -> Was Neutral:         {bull_neutral_pct:.2f}% ({total_bull_obs_neutral_trade} days)")

        # --- Generate Pie Chart for Bullish Observation ---
        try:
            labels = 'Continued Bullish', 'Reversed to Bearish', 'Neutral Trading Window'
            sizes = [total_bull_obs_continue_bull_trade, total_bull_obs_reverse_bear_trade, total_bull_obs_neutral_trade]
            # Ensure sizes correspond to labels and filter out zero-sized slices if you prefer cleaner charts
            valid_indices = [i for i, size in enumerate(sizes) if size > 0]
            labels = [labels[i] for i in valid_indices]
            sizes = [sizes[i] for i in valid_indices]

            if sizes: # Only plot if there's data to show
                colors = ['#66b3ff', '#ff9999', '#99ff99'] # Blue, Red, Green (adjust if filtering)
                colors = [colors[i] for i in valid_indices] # Filter colors accordingly
                explode = [0.05 if i == 0 else 0 for i in range(len(sizes))] # Explode the first slice (Continued Bullish if present)


                fig_bull, ax_bull = plt.subplots()
                ax_bull.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                ax_bull.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                plt.title(f'Trading Window ({input_trade_start_time}-{input_trade_end_time}) Outcome Following BULLISH Observation ({input_obs_start_time}-{input_obs_end_time})\nTotal Bullish Observation Days: {total_valid_bullish_obs_days}')
                plt.tight_layout() # Adjust layout to prevent labels overlapping
                print("\nDisplaying pie chart for Bullish Observation days...")
                plt.show() # Display the plot
            else:
                print("No data to plot for Bullish Observation outcomes.")

        except Exception as e:
            print(f"\nAn error occurred while generating the Bullish Observation pie chart: {e}")


    # --- Summary & Pie Chart for BEARISH Observation Window Days ---
    print(f"\n--- Analysis for days starting with a BEARISH Observation Window ({input_obs_start_time}-{input_obs_end_time}) ---")
    # Check if there were any days with a valid bearish observation window.
    if total_valid_bearish_obs_days == 0:
        print("No days found with a valid bearish observation window.")
    else:
        # Calculate percentages.
        bear_continue_pct = (total_bear_obs_continue_bear_trade / total_valid_bearish_obs_days) * 100
        bear_reverse_pct = (total_bear_obs_reverse_bull_trade / total_valid_bearish_obs_days) * 100
        bear_neutral_pct = (total_bear_obs_neutral_trade / total_valid_bearish_obs_days) * 100

        # Print summary statistics.
        print(f"Total days with Bearish observation window: {total_valid_bearish_obs_days}")
        print(f"  Outcome during Trading Window ({input_trade_start_time}-{input_trade_end_time}):")
        print(f"    -> Continued Bearish: {bear_continue_pct:.2f}% ({total_bear_obs_continue_bear_trade} days)")
        print(f"    -> Reversed to Bullish: {bear_reverse_pct:.2f}% ({total_bear_obs_reverse_bull_trade} days)")
        print(f"    -> Was Neutral:         {bear_neutral_pct:.2f}% ({total_bear_obs_neutral_trade} days)")

        # --- Generate Pie Chart for Bearish Observation ---
        try:
            labels = 'Continued Bearish', 'Reversed to Bullish', 'Neutral Trading Window'
            sizes = [total_bear_obs_continue_bear_trade, total_bear_obs_reverse_bull_trade, total_bear_obs_neutral_trade]
            # Ensure sizes correspond to labels and filter out zero-sized slices
            valid_indices = [i for i, size in enumerate(sizes) if size > 0]
            labels = [labels[i] for i in valid_indices]
            sizes = [sizes[i] for i in valid_indices]

            if sizes: # Only plot if there's data to show
                colors = ['#ff9999', '#66b3ff', '#99ff99'] # Red, Blue, Green (adjust if filtering)
                colors = [colors[i] for i in valid_indices] # Filter colors accordingly
                explode = [0.05 if i == 0 else 0 for i in range(len(sizes))] # Explode the first slice (Continued Bearish if present)

                fig_bear, ax_bear = plt.subplots()
                ax_bear.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                ax_bear.axis('equal')
                plt.title(f'Trading Window ({input_trade_start_time}-{input_trade_end_time}) Outcome Following BEARISH Observation ({input_obs_start_time}-{input_obs_end_time})\nTotal Bearish Observation Days: {total_valid_bearish_obs_days}')
                plt.tight_layout()
                print("\nDisplaying pie chart for Bearish Observation days...")
                plt.show() # Display the plot
            else:
                print("No data to plot for Bearish Observation outcomes.")

        except Exception as e:
            print(f"\nAn error occurred while generating the Bearish Observation pie chart: {e}")

    print("\nAnalysis complete.")
# --- End Main Execution Block ---