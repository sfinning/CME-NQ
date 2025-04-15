import pandas as pd

# Updated function to calculate both probabilities
def analyze_ohlc_probability_all_symbols(csv_filepath_or_url,
                                         timestamp_col='ts_event',
                                         open_col='open',
                                         high_col='high',
                                         low_col='low',
                                         symbol_col='symbol'):
    """
    Analyzes hourly OHLC data for ALL instruments to find hourly probabilities for:
    1. High Reversal: Trading > prev high AND <= current open within the hour.
    2. Low Reversal: Trading < prev low AND >= current open within the hour.

    Args:
        csv_filepath_or_url (str): Path/URL to the CSV file.
        timestamp_col (str): Timestamp column name (epoch ns).
        open_col (str): Open price column name.
        high_col (str): High price column name.
        low_col (str): Low price column name.
        symbol_col (str): Instrument symbol column name.

    Returns:
        dict: A dictionary where keys are symbol names (str) and values are
              pandas DataFrames. Each DataFrame has hours (0-23) as index
              and columns 'prob_high_reversal' and 'prob_low_reversal'.
              Returns an empty dictionary on major errors or no valid results.
    """
    all_results = {}
    try:
        # --- 1. Load Data ---
        print(f"Loading data from: {csv_filepath_or_url}")
        df = pd.read_csv(csv_filepath_or_url)
        print(f"Data loaded successfully. Found {len(df)} rows initially.")
        print(f"Columns found: {df.columns.tolist()}")

        # --- 2. Basic Validation ---
        required_cols = [timestamp_col, open_col, high_col, low_col, symbol_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing one or more required columns: {missing}")
            return all_results

        # --- 3. Identify Symbols ---
        unique_symbols = df[symbol_col].unique()
        if len(unique_symbols) == 0:
             print(f"Error: No symbols found in the '{symbol_col}' column.")
             return all_results
        print(f"Found symbols to analyze: {unique_symbols.tolist()}")

        # --- 4. Analyze Each Symbol ---
        for symbol in unique_symbols:
            print(f"\n--- Analyzing Symbol: {symbol} ---")
            try:
                # --- 4a. Filter Data ---
                df_symbol = df[df[symbol_col] == symbol].copy()
                print(f"Found {len(df_symbol)} rows for symbol {symbol}.")

                # Need at least 2 rows for shift()
                if len(df_symbol) < 2:
                     print(f"Skipping symbol '{symbol}': Insufficient data (< 2 rows).")
                     continue

                # --- 4b. Prepare Data (Timestamp, Sort, Shift) ---
                try:
                    df_symbol['datetime'] = pd.to_datetime(df_symbol[timestamp_col], unit='ns', errors='coerce')
                except ValueError as e:
                    print(f"Warning: Error converting timestamp for '{symbol}': {e}. Skipping.")
                    continue

                df_symbol = df_symbol.dropna(subset=['datetime'])
                if len(df_symbol) < 2:
                    print(f"Skipping symbol '{symbol}': Insufficient valid timestamp data (< 2 rows).")
                    continue

                df_symbol = df_symbol.sort_values(by='datetime').reset_index(drop=True)
                df_symbol['hour'] = df_symbol['datetime'].dt.hour

                # Get previous high AND low
                df_symbol['previous_high'] = df_symbol[high_col].shift(1)
                df_symbol['previous_low'] = df_symbol[low_col].shift(1) # Added previous low

                # Drop rows with NaN in previous high/low & check remaining data
                df_symbol = df_symbol.dropna(subset=['previous_high', 'previous_low'])
                if df_symbol.empty:
                    print(f"Skipping symbol '{symbol}': No data after handling missing previous high/low.")
                    continue
                print(f"Data preparation complete for {symbol}.")

                # --- 4c. Identify Events ---
                # Event 1: Trade > prev high, then <= current open
                df_symbol['event_high_reversal'] = (df_symbol[high_col] > df_symbol['previous_high']) & \
                                                   (df_symbol[low_col] <= df_symbol[open_col])

                # Event 2: Trade < prev low, then >= current open
                df_symbol['event_low_reversal'] = (df_symbol[low_col] < df_symbol['previous_low']) & \
                                                  (df_symbol[high_col] >= df_symbol[open_col]) # Added low reversal event

                print(f"Event identification complete for {symbol}.")

                # --- 4d. Calculate Probabilities ---
                print(f"Calculating probabilities for {symbol}...")
                hourly_counts = df_symbol.groupby('hour').size().reindex(range(24), fill_value=0)

                # Counts for High Reversal
                event_counts_high = df_symbol[df_symbol['event_high_reversal']].groupby('hour').size().reindex(range(24), fill_value=0)
                # Counts for Low Reversal
                event_counts_low = df_symbol[df_symbol['event_low_reversal']].groupby('hour').size().reindex(range(24), fill_value=0) # Added low counts

                # Calculate probabilities, avoid division by zero
                probabilities_high = pd.Series(0.0, index=range(24))
                probabilities_low = pd.Series(0.0, index=range(24)) # Added low probs

                non_zero_hours = hourly_counts[hourly_counts > 0].index
                if not non_zero_hours.empty:
                    probabilities_high.loc[non_zero_hours] = event_counts_high.loc[non_zero_hours] / hourly_counts.loc[non_zero_hours]
                    probabilities_low.loc[non_zero_hours] = event_counts_low.loc[non_zero_hours] / hourly_counts.loc[non_zero_hours] # Calc low probs

                # --- 4e. Store Results for Symbol ---
                # Store both probabilities in a DataFrame for this symbol
                symbol_probs = pd.DataFrame({
                    'prob_high_reversal': probabilities_high,
                    'prob_low_reversal': probabilities_low # Added low probs column
                }, index=range(24))

                all_results[symbol] = symbol_probs # Store the DataFrame
                print(f"Probability calculation complete for {symbol}.")

            except Exception as e:
                print(f"Warning: An unexpected error occurred while processing symbol '{symbol}': {e}")
                print(f"Skipping analysis for symbol '{symbol}'.")
                continue

        # --- 5. Return All Results ---
        return all_results # Returns dict {symbol: DataFrame}

    # Error handling remains the same...
    except FileNotFoundError:
        print(f"Error: File/URL not found at {csv_filepath_or_url}")
        return {}
    except KeyError as e:
        print(f"Error: Column not found - {e}. Check column name parameters.")
        return {}
    except pd.errors.EmptyDataError:
        print(f"Error: No data found in the file/URL: {csv_filepath_or_url}")
        return {}
    except Exception as e:
        print(f"An critical error occurred during initial loading or setup: {e}")
        return {}


# --- How to Use ---

# 1. Set the URL
file_url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1h.csv'

# 2. Run the analysis
# results_dict now contains {symbol: DataFrame(index=hour, columns=[prob_high_reversal, prob_low_reversal])}
results_dict = analyze_ohlc_probability_all_symbols(
    csv_filepath_or_url=file_url,
    timestamp_col='ts_event',
    open_col='open',
    high_col='high',
    low_col='low',
    symbol_col='symbol'
)

# 3. Combine results, calculate averages, and print for BOTH scenarios
if results_dict:
    print("\n--- Combining Results ---")
    try:
        # Create separate combined DataFrames for each probability type
        combined_df_high = pd.DataFrame({symbol: df['prob_high_reversal']
                                         for symbol, df in results_dict.items()})
        combined_df_low = pd.DataFrame({symbol: df['prob_low_reversal']
                                        for symbol, df in results_dict.items()})

        # Sort columns alphabetically
        combined_df_high = combined_df_high.sort_index(axis=1)
        combined_df_low = combined_df_low.sort_index(axis=1)

        # --- Display High Reversal Results ---
        print("\n--- Combined Analysis Results: High Reversal ---")
        print("Probability of trading > prev high AND <= current open, by hour (UTC):")
        print("-" * 70)
        try: formatted_df_high = combined_df_high.map(lambda x: f"{x:.2%}")
        except AttributeError: formatted_df_high = combined_df_high.applymap(lambda x: f"{x:.2%}")
        print(formatted_df_high)
        print("-" * 70)

        if not combined_df_high.empty:
            average_probabilities_high = combined_df_high.mean(axis=1)
            print("\n--- Average High Reversal Probabilities Across All Symbols ---")
            print("Average probability for each hour (UTC):")
            print("-" * 70)
            try: print(average_probabilities_high.map(lambda x: f"{x:.2%}"))
            except AttributeError: print(average_probabilities_high.apply(lambda x: f"{x:.2%}"))
            print("-" * 70)

        # --- Display Low Reversal Results ---
        print("\n--- Combined Analysis Results: Low Reversal ---")
        print("Probability of trading < prev low AND >= current open, by hour (UTC):")
        print("-" * 70)
        try: formatted_df_low = combined_df_low.map(lambda x: f"{x:.2%}")
        except AttributeError: formatted_df_low = combined_df_low.applymap(lambda x: f"{x:.2%}")
        print(formatted_df_low)
        print("-" * 70)

        if not combined_df_low.empty:
            average_probabilities_low = combined_df_low.mean(axis=1)
            print("\n--- Average Low Reversal Probabilities Across All Symbols ---")
            print("Average probability for each hour (UTC):")
            print("-" * 70)
            try: print(average_probabilities_low.map(lambda x: f"{x:.2%}"))
            except AttributeError: print(average_probabilities_low.apply(lambda x: f"{x:.2%}"))
            print("-" * 70)
        # --------------------------------------

        print("\nAnalysis finished.")

    except Exception as e:
         print(f"\nError combining, averaging, or printing results: {e}")

else:
    print("\nAnalysis could not be completed or no valid symbol data found to combine.")