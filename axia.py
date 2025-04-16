# -*- coding: utf-8 -*-
import pandas as pd
from datetime import time, datetime, timedelta, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import sys

# --- Function to Analyze Dynamic Range Breakout and Return to Range End Open ---
# Corrected version: Removed unnecessary try-except around sequence increment
def analyze_dynamic_range_rto(df, range_start_str, range_end_str, post_range_end_str, target_day_name, user_timezone):
    """
    Analyzes price action based on a user-defined initial range within a potentially
    date-filtered and timezone-converted DataFrame. Optionally filters by day of week.
    Processes ALL matching days found.

    Assumes df.index is already localized to user_timezone and potentially date-filtered.

    Returns: (matching_days_processed, successful_rto_sequences)
    """
    day_description = target_day_name + "s" if target_day_name else "all days"
    print(f"\nAnalyzing Dynamic Range RTO sequence for {day_description}")
    print(f"Timezone for Analysis: {user_timezone.key}")
    print(f"Initial Range Window: {range_start_str} - <{range_end_str} ({user_timezone.key})")
    print(f"RTO Reference Price: Open at {range_end_str} bar ({user_timezone.key})")
    print(f"Post-Range Window: {range_end_str} - <{post_range_end_str} ({user_timezone.key})")

    matching_days_processed = 0
    successful_sequences = 0 # Initialized correctly

    # Parse time strings into time objects
    try:
        range_start_t = time.fromisoformat(range_start_str)
        range_end_t = time.fromisoformat(range_end_str)
        post_range_end_t = time.fromisoformat(post_range_end_str)
    except (ValueError, TypeError):
        print("\nError: Invalid time format/inputs passed to analysis function. Cannot proceed.")
        # Return 0 counts as analysis cannot run with invalid times
        return 0, 0

    # Filter data for the target day IF specified
    if target_day_name:
        data_to_process = df[df.index.day_name() == target_day_name]
        if data_to_process.empty:
            print(f"No data found for {target_day_name} within the specified date range/timezone.")
            return 0, 0
    else:
        data_to_process = df
        if data_to_process.empty:
            print(f"Input dataframe (post-date-filter) is empty. Cannot process.")
            return 0, 0

    unique_days_in_data = data_to_process.index.normalize().unique()
    total_days_available_full = len(unique_days_in_data)

    if total_days_available_full > 0:
         print(f"Found {total_days_available_full} unique day(s) matching criteria within date range. Processing all.")
    # No message needed if 0 days found, handled later

    # Process ALL relevant days found
    for day_date in unique_days_in_data:
        day_data = data_to_process[data_to_process.index.date == day_date.date()]
        if day_data.empty: continue # Skip if somehow day data is empty

        current_day_date_part = day_date.date()
        try:
            # Define precise datetime boundaries for the analysis windows using user's timezone
            range_start_dt = datetime.combine(current_day_date_part, range_start_t, tzinfo=user_timezone)
            range_end_dt = datetime.combine(current_day_date_part, range_end_t, tzinfo=user_timezone)
            post_range_end_dt = datetime.combine(current_day_date_part, post_range_end_t, tzinfo=user_timezone)

            # Define exclusive end times (important for slicing)
            range_end_exclusive_dt = range_end_dt - timedelta(microseconds=1)
            post_range_end_exclusive_dt = post_range_end_dt - timedelta(microseconds=1)

        except Exception as e:
            print(f"Warning: Could not create window boundary for {current_day_date_part} in {user_timezone.key}. Skipping day. Error: {e}")
            continue

        # --- 1. Analyze Initial Range ---
        initial_range_data = day_data.loc[range_start_dt:range_end_exclusive_dt]
        if initial_range_data.empty:
            continue # Skip if no data falls within the time window on this day

        matching_days_processed += 1 # Count day only if it has data in the initial range
        range_high = initial_range_data['High'].max()
        range_low = initial_range_data['Low'].min()

        # --- 2. Get Reference Price & Post-Range Data ---
        post_range_data = day_data.loc[range_end_dt:post_range_end_exclusive_dt]
        if post_range_data.empty:
            continue # Skip if no data after range end
        try:
            # Get open price of the *first* bar at or after the range end time
            reference_open_price = post_range_data['Open'].iloc[0]
        except IndexError:
            continue # Skip if can't find the reference open price

        # --- 3. Find First Breakout after Range End Time ---
        first_breakout_time = None
        # Find potential first high break time
        high_break_condition = post_range_data['High'] > range_high
        first_high_break_time = post_range_data[high_break_condition].index.min() if high_break_condition.any() else None
        # Find potential first low break time
        low_break_condition = post_range_data['Low'] < range_low
        first_low_break_time = post_range_data[low_break_condition].index.min() if low_break_condition.any() else None

        # Determine the *actual* first breakout
        if first_high_break_time and first_low_break_time:
            first_breakout_time = min(first_high_break_time, first_low_break_time)
        elif first_high_break_time:
            first_breakout_time = first_high_break_time
        elif first_low_break_time:
            first_breakout_time = first_low_break_time

        if first_breakout_time is None:
            continue # Skip RTO check if no breakout happened before post_range_end_dt

        # --- 4. Check for Return to Open (RTO) *after* Breakout and *before* Post-Range End Time ---
        # Select data strictly *after* the breakout timestamp up to the end of the check window
        rto_check_data = post_range_data[post_range_data.index > first_breakout_time]

        if not rto_check_data.empty:
            # Condition: Price touches or crosses the reference open price
            rto_condition = (rto_check_data['Low'] <= reference_open_price) & (rto_check_data['High'] >= reference_open_price)

            if rto_condition.any():
                 # Increment count if RTO condition met after breakout within window
                 successful_sequences += 1 # Corrected: No unnecessary try-except

    print(f"\nAnalysis complete.")
    # Provide message only if days were found but none met the inner criteria
    if matching_days_processed == 0 and total_days_available_full > 0:
        print(f"No days within the filtered criteria had data in the initial range window {range_start_str}-<{range_end_str} ({user_timezone.key}).")
    # If total_days_available_full was 0, prior messages handled it.

    return matching_days_processed, successful_sequences


# --- Helper Function for Time Bucket Labels (Unused in this analysis) ---
def get_bucket_label(bucket_index, interval_minutes=12):
    """Calculates a time bucket label string."""
    start_minute_of_day = bucket_index * interval_minutes
    end_minute_of_day = min(start_minute_of_day + interval_minutes - 1, 1439) # 1439 = 23:59
    start_h, start_m = divmod(start_minute_of_day, 60)
    end_h, end_m = divmod(end_minute_of_day, 60)
    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"

# --- Function to get and validate user input ---
def get_user_input():
    """
    Prompts user for Time Windows, Timezone, optionally Day of Week,
    and optionally Start/End Dates.
    Returns time strings, target_day_name (str or None), timezone name,
    and start/end date strings (str or None).
    """
    day_number_map = { 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday" }
    allowed_nums_str = ", ".join(map(str, day_number_map.keys()))
    timezone_shortcuts = {"EST": "America/New_York", "CST": "America/Chicago", "UTC": "UTC"}

    range_start_str, range_end_str, post_range_end_str = None, None, None
    target_day_name, target_tz_name = None, None
    start_date_str, end_date_str = None, None # Initialize date strings

    print("--- Enter Analysis Time Windows (Local Time) ---")
    # Get Time Windows
    while True:
        start_input = input("Enter Initial Range Start Time (HH:MM format, e.g., 06:00): ")
        try: time.fromisoformat(start_input); range_start_str = start_input; break
        except ValueError: print("Invalid time format. Please use HH:MM (24-hour).")
    while True:
        end_input = input(f"Enter Initial Range End Time (HH:MM format, AFTER {range_start_str}, e.g., 07:00): ")
        try:
            if time.fromisoformat(end_input) > time.fromisoformat(range_start_str): range_end_str = end_input; break
            else: print("Initial Range End Time must be after Start Time.")
        except ValueError: print("Invalid time format. Please use HH:MM (24-hour).")
    while True:
        post_end_input = input(f"Enter Post-Range End Time (HH:MM format, AFTER {range_end_str}, e.g., 08:00): ")
        try:
            if time.fromisoformat(post_end_input) > time.fromisoformat(range_end_str): post_range_end_str = post_end_input; break
            else: print("Post-Range End Time must be after Initial Range End Time.")
        except ValueError: print("Invalid time format. Please use HH:MM (24-hour).")

    # Get Timezone
    print("\n--- Enter Timezone ---")
    print("This determines the local time for the analysis windows.")
    print("Enter Olson timezone name or shortcut (EST, CST, UTC). Examples: America/New_York, Europe/London")
    while True:
        tz_input = input("Enter Timezone: ").strip()
        resolved_tz_name = None; tz_input_upper = tz_input.upper()
        if tz_input_upper in timezone_shortcuts:
            resolved_tz_name = timezone_shortcuts[tz_input_upper]; print(f"Shortcut '{tz_input}' mapped to '{resolved_tz_name}'.")
        else: resolved_tz_name = tz_input
        try:
            ZoneInfo(resolved_tz_name); target_tz_name = resolved_tz_name; print(f"Using timezone: {target_tz_name}"); break
        except ZoneInfoNotFoundError: print(f"Error: Timezone '{resolved_tz_name}' not found. Please use a valid Olson name or shortcut.")
        except Exception as e: print(f"An unexpected error occurred validating timezone '{resolved_tz_name}': {e}")

    # Optional Day of Week Filter
    print("\n--- Optional Day of Week Filter ---")
    while True:
        filter_choice = input("Filter by a specific day of the week? (yes/no): ").strip().lower()
        if filter_choice in ['yes', 'y']:
            while True:
                day_input_str = input(f"Enter the Day of the Week Number ({allowed_nums_str}): ")
                try:
                    day_num = int(day_input_str)
                    target_day_name = day_number_map[day_num] # Get name from map
                    print(f"Selected day: {target_day_name}")
                    break # Exit inner loop (day selection)
                except (ValueError, KeyError):
                    print(f"Invalid number. Please enter one of: {allowed_nums_str}.")
            break # Exit outer loop (yes/no choice)
        elif filter_choice in ['no', 'n']:
            target_day_name = None # Set to None to indicate no filter
            print("Analyzing all days of the week.")
            break # Exit outer loop (yes/no choice)
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    # Optional Date Range Filter
    print("\n--- Optional Date Range Filter ---")
    while True:
        date_filter_choice = input("Filter by a specific date range? (yes/no): ").strip().lower()
        if date_filter_choice in ['yes', 'y']:
            # Get Start Date
            while True:
                s_date_input = input("Enter Analysis Start Date (YYYY-MM-DD format): ")
                try:
                    start_dt_obj = datetime.strptime(s_date_input, '%Y-%m-%d').date() # Validate format
                    start_date_str = s_date_input # Store valid string
                    break
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD.")
            # Get End Date
            while True:
                e_date_input = input(f"Enter Analysis End Date (YYYY-MM-DD format, on or after {start_date_str}): ")
                try:
                    end_dt_obj = datetime.strptime(e_date_input, '%Y-%m-%d').date()
                    if end_dt_obj >= start_dt_obj:
                         end_date_str = e_date_input # Store valid string
                         break
                    else:
                        print("End Date must be the same as or after Start Date.")
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD.")
            print(f"Date range selected: {start_date_str} to {end_date_str}")
            break # Exit date filter loop
        elif date_filter_choice in ['no', 'n']:
            start_date_str, end_date_str = None, None # Explicitly set to None
            print("Analyzing all dates in the data.")
            break # Exit date filter loop
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    # Return all collected inputs
    return range_start_str, range_end_str, post_range_end_str, target_day_name, target_tz_name, start_date_str, end_date_str


# --- Main Script ---
# --- 1. Configuration and Data Loading ---
url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv'
master_df = pd.DataFrame() # Use a different name to avoid confusion with loop variables

try:
    print("Loading data...")
    # Load the raw data
    temp_df = pd.read_csv(url)
    if 'ts_event' not in temp_df.columns or 'symbol' not in temp_df.columns:
        raise ValueError("Essential columns 'ts_event' or 'symbol' missing.")

    # Convert timestamp and set index
    temp_df['ts_event'] = pd.to_datetime(temp_df['ts_event'], unit='ns')
    temp_df = temp_df.set_index('ts_event').sort_index()
    print(f"Data loaded. Index is initially timezone-naive. Range: {temp_df.index.min()} to {temp_df.index.max()}")

    # Select and rename OHLCV columns
    ohlcv_cols = {'open': 'Open','high': 'High','low': 'Low','close': 'Close','volume': 'Volume'}
    required_cols = list(ohlcv_cols.keys())
    if all(col in temp_df.columns for col in required_cols):
        # Select only needed columns and rename
        master_df = temp_df[required_cols].rename(columns=ohlcv_cols)
        # Convert OHLCV columns to numeric, coercing errors
        for col in master_df.columns: # Iterate over renamed columns
             if col != 'Volume': # Apply only to OHLC
                 master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
        # Handle Volume separately if needed, assume it might be int or float
        master_df['Volume'] = pd.to_numeric(master_df['Volume'], errors='coerce').fillna(0).astype(int) # Example: coerce, fill NaN, ensure int

        # Check for NaNs introduced in OHLC columns
        if master_df[['Open', 'High', 'Low', 'Close']].isnull().any().any():
            print("Warning: Some non-numeric OHLC data converted to NaN during preparation.")
            # Optionally, decide whether to drop rows with NaN in OHLC
            # master_df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            # print("Rows with NaN in OHLC columns were dropped.")

        print("OHLCV data preparation complete.")
    else:
        missing = [c for c in required_cols if c not in temp_df.columns]
        raise ValueError(f"Missing required OHLCV columns: {missing}")

except Exception as e:
    print(f"An error occurred during data loading/preparation: {e}")
    master_df = pd.DataFrame() # Ensure it's empty on error

# --- 2. Get User Input ---
user_timezone = None
target_day_name = None
start_date_str, end_date_str = None, None
range_start_str, range_end_str, post_range_end_str = None, None, None
analysis_possible = False # Flag to track if we should proceed

if not master_df.empty:
    # Get all inputs from the user
    range_start_str, range_end_str, post_range_end_str, target_day_name, target_tz_name, start_date_str, end_date_str = get_user_input()
    # Validate essential inputs received for analysis setup
    if target_tz_name and range_start_str and range_end_str and post_range_end_str:
        try:
            user_timezone = ZoneInfo(target_tz_name)
            # If timezone is valid, we can potentially proceed
            analysis_possible = True
        except Exception as e:
            print(f"\nFATAL ERROR creating timezone '{target_tz_name}': {e}")
            # analysis_possible remains False
    else:
        print("\nOne or more essential inputs (times, timezone) were not provided.")
        # analysis_possible remains False
else:
    print("\nSkipping user input due to data loading issues.")

# --- 3. Apply Date Filter (if provided and possible) ---
filtered_df = pd.DataFrame() # Initialize empty df
if analysis_possible:
    df_to_filter = master_df.copy() # Work on a copy

    if start_date_str and end_date_str:
        try:
            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            print(f"\nApplying date filter: {start_date_obj} to {end_date_obj} (inclusive) on naive index...")
            # Filter using the date part of the naive index
            filtered_df = df_to_filter[
                (df_to_filter.index.date >= start_date_obj) &
                (df_to_filter.index.date <= end_date_obj)
            ] # No need for .copy() here as it's a slice result
            if filtered_df.empty:
                 print("Warning: No data found within the specified date range. Analysis cannot proceed.")
                 analysis_possible = False # Cannot proceed if filter yields nothing
            else:
                 print(f"Data filtered by date. New range: {filtered_df.index.min()} to {filtered_df.index.max()}")
        except ValueError:
            print("Error: Invalid date format encountered during filtering. Analysis cannot proceed.")
            analysis_possible = False
    else:
        print("\nNo date filter applied.")
        filtered_df = df_to_filter # Use the full (copied) dataframe
        # Ensure it's not empty even without filter
        if filtered_df.empty:
             print("Error: Dataframe is empty even before date filtering. Analysis cannot proceed.")
             analysis_possible = False

# --- 4. Timezone Conversion & Analysis ---
# Initialize results vars outside try block
days_processed = 0
successful_sequences = 0
analysis_completed = False # Explicit flag

if analysis_possible and not filtered_df.empty:
    try:
        print(f"\nLocalizing index to UTC and converting to {user_timezone.key}...")
        # Work directly on filtered_df for timezone conversion
        filtered_df.index = filtered_df.index.tz_localize('UTC', ambiguous='infer').tz_convert(user_timezone)
        print("Timezone conversion complete.")

        # Perform the dynamic range RTO analysis using the timezone-aware df
        days_processed, successful_sequences = analyze_dynamic_range_rto(
            filtered_df, range_start_str, range_end_str, post_range_end_str,
            target_day_name, user_timezone
        )
        analysis_completed = True # Mark as successfully completed

    except Exception as e:
         # Catch errors during timezone conversion or the analysis function call
         print(f"\nAn error occurred during timezone conversion or analysis: {e}")
         # analysis_completed remains False, results vars keep initial values (0)

else:
     # Provide feedback if analysis didn't run due to earlier steps
     if not analysis_possible: pass # Reason already printed during input/setup
     elif filtered_df.empty: pass # Reason already printed during date filtering
     else: print("\nAnalysis skipped due to unknown reasons after input/filtering.")


# --- 5. Display Results ---
print("\n--- Dynamic Range Breakout & RTO Analysis Results ---")
# Display results only if analysis was attempted and successfully completed
if analysis_completed:
    # Prepare descriptions for display
    day_analyzed_desc = target_day_name if target_day_name else "All Days"
    day_processed_desc = f"'{target_day_name}' days" if target_day_name else "days"
    date_range_desc = f"{start_date_str} to {end_date_str}" if start_date_str and end_date_str else "All Dates"

    # Display parameters used
    print(f"Date Range Analyzed:      {date_range_desc}")
    print(f"Days Analyzed:            {day_analyzed_desc}")
    print(f"Timezone:                 {user_timezone.key}")
    print(f"Initial Range Window:     {range_start_str} - <{range_end_str} (Local Time)")
    print(f"RTO Reference Price:      Open at {range_end_str} bar (Local Time)")
    print(f"Post-Range Window:        {range_end_str} - <{post_range_end_str} (Local Time)")
    print("-" * 60)

    # Display calculated results
    if days_processed > 0:
        print(f"Total {day_processed_desc} processed (within date range) with data in {range_start_str}-<{range_end_str} range: {days_processed}")
        print(f"Days with sequence (Range -> Breakout -> RTO to {range_end_str} Open before {post_range_end_str}): {successful_sequences}")
        probability = (successful_sequences / days_processed) * 100 # Avoid division by zero checked by days_processed > 0
        print(f"Probability of sequence occurring on qualifying days: {probability:.2f}%")
    else:
        # Handle case where analysis ran but found no days meeting all criteria
        print("No qualifying days found or processed successfully within the criteria.")
        print("(This could be due to date/day filters or no data in the specified time windows on matching days).")

    print("-" * 60)
else:
    # Message if analysis did not complete successfully
    print("Analysis could not be completed.")
    print("Please check input parameters, data availability, and console messages for specific errors.")


# --- How to Run ---
# 1. Make sure you have pandas and tzdata installed: `pip install pandas tzdata`
# 2. Save the script as a Python file (e.g., `full_analyzer.py`).
# 3. Run from your terminal: `python full_analyzer.py`
# 4. Follow the prompts for times, timezone, day filter, and date filter.