# -*- coding: utf-8 -*-
import pandas as pd
# Import necessary components from datetime, including the specific error types
from datetime import time, datetime, timedelta, date
# Import necessary components from zoneinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import sys
import math
from pytz import AmbiguousTimeError, NonExistentTimeError # Import math library, potentially for future use (not currently used)

# --- Function to Analyze Dynamic Range Breakout and Return to Range End Open ---
# Tracks RTO events, conditioned on Pre-Breakout RTO status, including interval distribution.
# MODIFIED Post-RTO check to be strictly AFTER breakout time.
def analyze_dynamic_range_rto(df, range_start_str, range_end_str, post_range_end_str, target_day_name, user_timezone):
    """
    Analyzes price action, tracking RTO events.
    MODIFIED: Tracks Post-Breakout RTO interval occurrences separately based on
    whether a Pre-Breakout RTO occurred on the same day.
    MODIFIED: Post-Breakout RTO search now starts strictly AFTER the breakout bar.

    For each relevant day:
    1. Finds High/Low of initial range.
    2. Gets Open price at range_end_str (Reference Price).
    3. Finds first breakout after range_end_str.
    4. If breakout occurs:
        a. Checks if an RTO occurred between range_end_str and the breakout time (Pre-Breakout RTO).
        b. Finds the first RTO strictly after the breakout time but before post_range_end_str (Post-Breakout RTO).
        c. If Post-Breakout RTO occurs, determines which of 5 intervals it falls into
           and increments the appropriate conditional interval counter.
    5. Counts days based on Breakout, Pre-RTO, and Post-RTO status.

    Returns: (matching_days_processed,
              breakout_days_count,           # Total days with any breakout
              breakout_with_pre_rto_count,   # Denom for P(PostRTO | PreRTO)
              post_rto_given_pre_rto_count,  # Numerator for P(PostRTO | PreRTO)
              breakout_no_pre_rto_count,     # Denom for P(PostRTO | NoPreRTO)
              post_rto_given_no_pre_rto_count,# Numerator for P(PostRTO | NoPreRTO)
              rto_intervals_given_pre_rto,   # Interval counts for days WITH PreRTO
              rto_intervals_given_no_pre_rto # Interval counts for days with NO PreRTO
             )
    """
    # --- Initial setup and parameter display ---
    day_description = target_day_name + "s" if target_day_name else "all applicable days"
    print(f"\nAnalyzing Dynamic Range RTO sequence for {day_description}")
    print(f"Timezone for Analysis: {user_timezone.key}")
    print(f"Initial Range Window: {range_start_str} - <{range_end_str} ({user_timezone.key})")
    print(f"RTO Reference Price: Open at {range_end_str} bar ({user_timezone.key})")
    print(f"Post-Range Window: {range_end_str} - <{post_range_end_str} ({user_timezone.key})")

    # --- Initialize Counters ---
    matching_days_processed = 0
    breakout_days_count = 0
    breakout_with_pre_rto_count = 0
    post_rto_given_pre_rto_count = 0
    breakout_no_pre_rto_count = 0
    post_rto_given_no_pre_rto_count = 0
    rto_intervals_given_pre_rto = [0] * 5
    rto_intervals_given_no_pre_rto = [0] * 5
    # ---

    # --- Calculate interval durations ---
    try:
        # Parse time strings into time objects
        range_start_t = time.fromisoformat(range_start_str)
        range_end_t = time.fromisoformat(range_end_str)
        post_range_end_t = time.fromisoformat(post_range_end_str)

        # Use a dummy date to calculate timedelta between times
        dummy_date = date.min
        start_dt_dummy = datetime.combine(dummy_date, range_end_t)
        end_dt_dummy = datetime.combine(dummy_date, post_range_end_t)
        total_post_range_duration = end_dt_dummy - start_dt_dummy

        # Ensure the post-range window has a positive duration
        if total_post_range_duration <= timedelta(0):
             print("\nError: Post-Range Window duration must be positive. Cannot proceed.")
             # Return zeros for all counts including new interval lists
             return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5

        # Calculate the duration of each of the 5 intervals
        interval_duration = total_post_range_duration / 5
        print(f"Post-Range Duration: {total_post_range_duration}, Interval Duration: {interval_duration}")

    except (ValueError, TypeError) as e:
        print(f"\nError: Invalid time format/inputs ({e}). Cannot calculate intervals.")
        # Return zeros for all counts including new interval lists
        return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5

    # --- Filter data by day name if specified ---
    if target_day_name:
        data_to_process = df[df.index.day_name() == target_day_name]
        # Return early if no data matches the day filter
        if data_to_process.empty:
            print(f"No data found for the specified day: {target_day_name}")
            return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5
    else:
        data_to_process = df
        # Return early if the initial dataframe passed is empty
        if data_to_process.empty:
            print("Input DataFrame is empty. Cannot process.")
            return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5

    # Get unique days present in the data to iterate over
    unique_days_in_data = data_to_process.index.normalize().unique()
    total_days_available_full = len(unique_days_in_data)

    if total_days_available_full > 0:
         print(f"Found {total_days_available_full} unique day(s) matching criteria. Processing...")
    else:
         print("No unique days found in the data matching criteria.")
         return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5 # Return zeros if no days to process

    # --- Process each unique day ---
    for day_date in unique_days_in_data:
        # Select data only for the current day being processed
        day_data = data_to_process[data_to_process.index.date == day_date.date()]
        if day_data.empty: continue # Should not happen if unique_days come from data_to_process, but safety check

        current_day_date_part = day_date.date()
        interval_boundaries_day = [] # Define outside try block
        try:
            # Define precise datetime boundaries for the current day using user inputs and timezone
            range_start_dt = datetime.combine(current_day_date_part, range_start_t, tzinfo=user_timezone)
            range_end_dt = datetime.combine(current_day_date_part, range_end_t, tzinfo=user_timezone)
            post_range_end_dt_day = datetime.combine(current_day_date_part, post_range_end_t, tzinfo=user_timezone)

            # Define exclusive end times for slicing using .loc (subtract minimal duration)
            range_end_exclusive_dt = range_end_dt - timedelta(microseconds=1)
            post_range_end_exclusive_dt_day = post_range_end_dt_day - timedelta(microseconds=1)

            # Calculate interval start/end datetime boundaries for THIS specific day
            current_boundary_dt = range_end_dt # First interval starts at range_end_dt
            for i in range(5):
                end_dt_calc = current_boundary_dt + interval_duration
                # Ensure calculated end does not exceed the overall post-range end
                next_boundary_dt = min(end_dt_calc, post_range_end_dt_day)
                # Force the last interval's end to be exactly the post_range_end_dt_day
                if i == 4: next_boundary_dt = post_range_end_dt_day
                interval_boundaries_day.append((current_boundary_dt, next_boundary_dt))
                current_boundary_dt = next_boundary_dt # Start of next interval is end of current one

        except Exception as e:
            # Warn if boundaries can't be created for a day, then skip it
            print(f"Warning: Could not create window/interval boundary for {current_day_date_part}. Skipping day. Error: {e}")
            continue

        # --- 1. Analyze Initial Range ---
        # Select data within the initial range window [start, end)
        initial_range_data = day_data.loc[range_start_dt:range_end_exclusive_dt]
        # If no data exists in the initial range for this day, skip it
        if initial_range_data.empty: continue
        matching_days_processed += 1 # Increment count of days with data in the initial range

        # Find the highest high and lowest low within the initial range
        range_high = initial_range_data['High'].max()
        range_low = initial_range_data['Low'].min()

        # --- 2. Get Reference Price & Post-Range Data ---
        # Select data within the post-range window [start, end)
        post_range_data = day_data.loc[range_end_dt:post_range_end_exclusive_dt_day]
        # If no data exists in the post-range window, skip the day's breakout/RTO analysis
        if post_range_data.empty: continue

        try:
            # The reference price is the Open of the *first* bar within the post_range_data
            reference_open_price = post_range_data['Open'].iloc[0]
        except IndexError:
            # If there's no bar exactly at range_end_dt (e.g., data gap), skip the day
            continue

        # --- 3. Find First Breakout ---
        first_breakout_time = None # Initialize breakout time as None
        # Check for high breakout: High price exceeding the initial range high
        high_break_condition = post_range_data['High'] > range_high
        first_high_break_time = post_range_data[high_break_condition].index.min() if high_break_condition.any() else None

        # Check for low breakout: Low price dropping below the initial range low
        low_break_condition = post_range_data['Low'] < range_low
        first_low_break_time = post_range_data[low_break_condition].index.min() if low_break_condition.any() else None

        # Determine the actual first breakout time (earliest of high or low break)
        if first_high_break_time and first_low_break_time:
            first_breakout_time = min(first_high_break_time, first_low_break_time)
        elif first_high_break_time:
            first_breakout_time = first_high_break_time
        elif first_low_break_time:
            first_breakout_time = first_low_break_time
        # If neither occurred, first_breakout_time remains None

        # --- 4. Check for Pre-RTO, Post-RTO & Determine Interval (Only if Breakout Occurred) ---
        if first_breakout_time:
            breakout_days_count += 1 # Increment total days where a breakout was found
            pre_rto_occurred = False # Flag for Pre-Breakout RTO status for this day
            post_rto_occurred = False # Flag for Post-Breakout RTO status for this day
            first_post_rto_time = None # Timestamp of the first Post-Breakout RTO

            # --- 4a. Check for Pre-Breakout RTO ---
            # Check interval BEFORE the breakout bar
            pre_breakout_check_data = post_range_data[post_range_data.index < first_breakout_time]
            # Check if the RTO condition (Low <= Ref Price <= High) occurred in this pre-breakout data
            if not pre_breakout_check_data.empty:
                pre_rto_condition = (pre_breakout_check_data['Low'] <= reference_open_price) & (pre_breakout_check_data['High'] >= reference_open_price)
                if pre_rto_condition.any():
                    pre_rto_occurred = True
                    breakout_with_pre_rto_count += 1 # Increment count for breakout days WITH Pre-RTO
                else:
                    # Breakout occurred, but no RTO before it
                    breakout_no_pre_rto_count += 1 # Increment count for breakout days with NO Pre-RTO
            else: # If pre_breakout_check_data is empty (breakout on first bar)
                # Thus, no Pre-Breakout RTO was possible.
                breakout_no_pre_rto_count += 1 # Increment count for breakout days with NO Pre-RTO


            # --- 4b. Check for Post-Breakout RTO ---
            # Check interval STRICTLY AFTER the breakout bar
            # ===> MODIFIED LINE <===
            rto_check_data = post_range_data[post_range_data.index > first_breakout_time]
            # ===> END MODIFIED LINE <===

            # Check if the RTO condition occurred in this post-breakout data
            if not rto_check_data.empty:
                post_rto_condition = (rto_check_data['Low'] <= reference_open_price) & (rto_check_data['High'] >= reference_open_price)
                if post_rto_condition.any():
                     try:
                         # Find the timestamp of the *first* bar meeting the condition AFTER the breakout bar
                         first_post_rto_time = rto_check_data[post_rto_condition].index.min()
                         post_rto_occurred = True
                     except ValueError: pass # Should not happen if .any() is true, but safeguard

            # --- 4c. Increment Conditional Overall Counters & Conditional Interval Counts ---
            if post_rto_occurred: # Only proceed if an RTO was found strictly AFTER breakout
                # Increment the appropriate overall conditional *numerator*
                if pre_rto_occurred:
                    post_rto_given_pre_rto_count += 1 # Numerator for P(Post | Pre)
                else:
                    post_rto_given_no_pre_rto_count += 1 # Numerator for P(Post | No Pre)

                # Assign the found RTO time to the correct interval
                if first_post_rto_time: # Ensure we have a valid timestamp
                    interval_found_index = -1 # Default if not found
                    for i in range(5):
                        start_interval, end_interval = interval_boundaries_day[i]
                        is_in_interval = False
                        # Check interval using >= start and <= end for last interval, < end for others
                        if i == 4: # Last interval check [start, end]
                           if first_post_rto_time >= start_interval and first_post_rto_time <= end_interval:
                                is_in_interval = True
                        else: # Check for intervals 0 through 3 [start, end)
                            if first_post_rto_time >= start_interval and first_post_rto_time < end_interval:
                                is_in_interval = True

                        # If the RTO time is in this interval, increment the correct conditional counter
                        if is_in_interval:
                            interval_found_index = i
                            if pre_rto_occurred:
                                rto_intervals_given_pre_rto[i] += 1
                            else:
                                rto_intervals_given_no_pre_rto[i] += 1
                            break # Found the interval, stop checking
        # --- End of processing for a single day ---

    # --- Analysis Loop Finished ---
    print(f"\nAnalysis complete.")
    if matching_days_processed == 0 and total_days_available_full > 0:
        print(f"Warning: Although {total_days_available_full} days matched filters, none had data in the initial range window {range_start_str}-<{range_end_str} ({user_timezone.key}).")

    # Return all calculated counts
    return (matching_days_processed, breakout_days_count,
            breakout_with_pre_rto_count, post_rto_given_pre_rto_count,
            breakout_no_pre_rto_count, post_rto_given_no_pre_rto_count,
            rto_intervals_given_pre_rto, rto_intervals_given_no_pre_rto)


# --- Helper Function for Time Bucket Labels (Currently Unused by main logic) ---
def get_bucket_label(bucket_index, interval_minutes=12):
    """Calculates a time bucket label string (e.g., '09:30-09:41')."""
    start_minute_of_day = bucket_index * interval_minutes
    # Ensure end minute doesn't exceed the last minute of the day (1439 = 23*60 + 59)
    end_minute_of_day = min(start_minute_of_day + interval_minutes - 1, 1439)
    # Calculate hours and minutes
    start_h, start_m = divmod(start_minute_of_day, 60)
    end_h, end_m = divmod(end_minute_of_day, 60)
    # Format as HH:MM-HH:MM
    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"


# --- Function to get and validate user input ---
def get_user_input():
    """
    Prompts user for Time Windows, Timezone, optionally Day of Week,
    and optionally Start/End Dates. Validates inputs.
    Returns: time strings, target_day_name (str or None), timezone name (str),
             and start/end date strings (str or None).
    """
    # Map numbers to weekday names for filtering convenience
    day_number_map = { 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday" }
    allowed_nums_str = ", ".join(map(str, day_number_map.keys()))
    # Common timezone shortcuts mapped to IANA names
    timezone_shortcuts = {"EST": "America/New_York", "CST": "America/Chicago", "PST": "America/Los_Angeles"}

    # Initialize variables to store user inputs
    range_start_str, range_end_str, post_range_end_str = None, None, None
    target_day_name, target_tz_name = None, None
    start_date_str, end_date_str = None, None

    print("--- Enter Analysis Time Windows (Local Time) ---")
    # Get Initial Range Start Time
    while True:
        start_input = input("Enter Initial Range Start Time (HH:MM format, e.g., 06:00): ").strip()
        try:
            time.fromisoformat(start_input) # Validate format
            range_start_str = start_input
            break
        except ValueError:
            print("Invalid time format. Please use HH:MM (24-hour).")

    # Get Initial Range End Time (must be after Start Time)
    while True:
        end_input = input(f"Enter Initial Range End Time (HH:MM format, AFTER {range_start_str}, e.g., 07:00): ").strip()
        try:
            t_end = time.fromisoformat(end_input)
            t_start = time.fromisoformat(range_start_str) # Re-parse start time for comparison
            if t_end > t_start:
                range_end_str = end_input
                break
            else:
                print("Initial Range End Time must be strictly after Start Time.")
        except ValueError:
            print("Invalid time format. Please use HH:MM (24-hour).")

    # Get Post-Range End Time (must be after Initial Range End Time)
    while True:
        post_end_input = input(f"Enter Post-Range End Time (HH:MM format, AFTER {range_end_str}, e.g., 08:00): ").strip()
        try:
            t_post_end = time.fromisoformat(post_end_input)
            t_range_end = time.fromisoformat(range_end_str) # Re-parse range end time
            if t_post_end > t_range_end:
                post_range_end_str = post_end_input
                break
            else:
                print("Post-Range End Time must be strictly after Initial Range End Time.")
        except ValueError:
            print("Invalid time format. Please use HH:MM (24-hour).")

    print("\n--- Enter Timezone ---")
    # Get Timezone (allow shortcuts or full IANA names)
    while True:
        tz_input = input("Enter Timezone (e.g., EST, CST, PST, UTC, America/New_York): ").strip()
        resolved_tz_name = None
        tz_input_upper = tz_input.upper()

        # Check if the input matches a defined shortcut
        if tz_input_upper in timezone_shortcuts:
            resolved_tz_name = timezone_shortcuts[tz_input_upper]
            print(f"Shortcut '{tz_input_upper}' mapped to '{resolved_tz_name}'.")
        else:
            # Assume the user entered a full IANA name (or potentially UTC)
            resolved_tz_name = tz_input

        # Validate the resolved timezone name using zoneinfo
        try:
            ZoneInfo(resolved_tz_name) # Attempt to create the ZoneInfo object
            target_tz_name = resolved_tz_name
            print(f"Using timezone: {target_tz_name}")
            break # Exit loop if timezone is valid
        except ZoneInfoNotFoundError:
            print(f"Error: Timezone '{resolved_tz_name}' not found.")
            print("Please use standard IANA names (e.g., America/New_York, Europe/London, UTC) or shortcuts (EST, CST, PST).")
        except Exception as e:
            # Catch any other unexpected errors during validation
            print(f"Unexpected error validating timezone '{resolved_tz_name}': {e}")

    print("\n--- Optional Day of Week Filter ---")
    # Get Optional Day of Week Filter
    while True:
        filter_choice = input("Filter by a specific day of the week (Mon-Fri)? (yes/no): ").strip().lower()
        if filter_choice in ['yes', 'y']:
            # Prompt for day number if filtering is chosen
            while True:
                day_input_str = input(f"Enter Day Number ({allowed_nums_str} for Mon-Fri): ")
                try:
                    day_num = int(day_input_str)
                    # Check if the number corresponds to a valid day in our map
                    if day_num in day_number_map:
                        target_day_name = day_number_map[day_num]
                        print(f"Selected day: {target_day_name}")
                        break # Exit inner loop (day number valid)
                    else:
                        print(f"Invalid number. Please enter one of: {allowed_nums_str}.")
                except ValueError:
                    # Handle cases where input is not an integer
                    print(f"Invalid input. Please enter a number: {allowed_nums_str}.")
            break # Exit outer loop (filter choice 'yes' handled)
        elif filter_choice in ['no', 'n']:
            target_day_name = None # No day filter applied
            print("Analyzing all applicable days of the week found in the data.")
            break # Exit outer loop
        else:
            # Prompt again if input is not 'yes' or 'no'
            print("Invalid input. Please enter 'yes' or 'no'.")

    print("\n--- Optional Date Range Filter ---")
    # Get Optional Date Range Filter
    while True:
        date_filter_choice = input("Filter by a specific date range? (yes/no): ").strip().lower()
        if date_filter_choice in ['yes', 'y']:
            # Get Start Date
            while True:
                s_date_input = input("Enter Analysis Start Date (YYYY-MM-DD format): ").strip()
                try:
                    # Validate format and convert to date object
                    start_dt_obj = datetime.strptime(s_date_input, '%Y-%m-%d').date()
                    start_date_str = s_date_input # Store valid string
                    break
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD.")
            # Get End Date (must be on or after Start Date)
            while True:
                e_date_input = input(f"Enter Analysis End Date (YYYY-MM-DD format, on or after {start_date_str}): ").strip()
                try:
                    end_dt_obj = datetime.strptime(e_date_input, '%Y-%m-%d').date()
                    # Re-parse start date string to ensure comparison is correct
                    start_dt_obj_check = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                    if end_dt_obj >= start_dt_obj_check:
                        end_date_str = e_date_input # Store valid string
                        break
                    else:
                        print("End Date must be the same as or after Start Date.")
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD.")
            print(f"Date range selected: {start_date_str} to {end_date_str}")
            break # Exit outer loop (filter choice 'yes' handled)
        elif date_filter_choice in ['no', 'n']:
            start_date_str, end_date_str = None, None # No date filter
            print("Analyzing all dates present in the data.")
            break # Exit outer loop
        else:
             # Prompt again if input is not 'yes' or 'no'
            print("Invalid input. Please enter 'yes' or 'no'.")

    # Return all collected user inputs
    return range_start_str, range_end_str, post_range_end_str, target_day_name, target_tz_name, start_date_str, end_date_str


# --- Helper function to format and display interval results (NO DEBUG) ---
def display_interval_results(title, interval_counts, denominator, range_end_str, post_range_end_str):
    """Helper function to print formatted interval counts and probabilities."""
    print(title)
    # Only calculate and print if the denominator (number of days in scenario) is positive
    if denominator > 0:
        try:
            # Calculate interval timings for display purposes
            range_end_t_obj = time.fromisoformat(range_end_str)
            post_range_end_t_obj = time.fromisoformat(post_range_end_str)
            dummy_date = date.min
            start_dt_dummy = datetime.combine(dummy_date, range_end_t_obj)
            end_dt_dummy = datetime.combine(dummy_date, post_range_end_t_obj)
            total_duration = end_dt_dummy - start_dt_dummy

            # Check duration again, although checked earlier, good practice here too
            if total_duration <= timedelta(0): raise ValueError("Post-range duration must be positive")

            interval_duration = total_duration / 5
            current_interval_start_dt = start_dt_dummy

            # Loop through the 5 intervals
            for i in range(5):
                # Calculate the end time for the current interval
                current_interval_end_dt = current_interval_start_dt + interval_duration
                # Ensure the last interval ends exactly at the post-range end time due to potential float division inaccuracies
                if i == 4:
                    current_interval_end_dt = end_dt_dummy

                # Format start and end times for display
                start_t_str = current_interval_start_dt.strftime('%H:%M:%S')
                end_t_str = current_interval_end_dt.strftime('%H:%M:%S')

                # Create the label, using "<=" for the end of the last interval
                interval_label = f"{start_t_str} - <{end_t_str}"
                if i == 4:
                    interval_label = f"{start_t_str} - <={end_t_str}" # Inclusive label for last interval end

                # Get the count for this interval from the provided list
                rto_count = interval_counts[i]
                # Calculate the probability based on the scenario's denominator
                probability = (rto_count / denominator) * 100
                # Print the formatted result for this interval
                print(f"   Interval {i+1} ({interval_label}): Count={rto_count}, Probability={probability:.2f}%")

                # Set the start time for the next interval calculation
                current_interval_start_dt = current_interval_end_dt

        except Exception as e:
            # Fallback display if interval time calculation or formatting fails
            print(f"\n   Error calculating/displaying interval times: {e}")
            print("   Displaying interval counts and probabilities without time labels:")
            for i in range(5):
                 rto_count = interval_counts[i]
                 # Ensure denominator check before division even in fallback
                 probability = (rto_count / denominator) * 100 if denominator > 0 else 0
                 print(f"   Interval {i+1}: Count={rto_count}, Probability={probability:.2f}%")
    else:
        # Message if the denominator is zero (no days in this scenario)
        print(f"   (Denominator is {denominator}, cannot calculate interval probabilities for this scenario)")
    print("-" * 60) # Print separator line after each interval breakdown


# --- Main Script Execution Block ---
if __name__ == "__main__":

    # --- 1. Configuration and Data Loading ---
    # URL of the CSV data file
    url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv'
    master_df = pd.DataFrame() # Initialize empty DataFrame to store loaded data

    try:
        print("Loading data...")
        # Attempt to read the CSV data from the URL into a pandas DataFrame
        temp_df = pd.read_csv(url)

        # --- Basic Column Validation ---
        # Check if essential timestamp and symbol columns exist
        if 'ts_event' not in temp_df.columns or 'symbol' not in temp_df.columns:
            raise ValueError("Essential columns 'ts_event' or 'symbol' missing in the CSV.")

        # --- Timestamp Conversion and Indexing ---
        # Convert nanosecond timestamp column ('ts_event') to datetime objects
        temp_df['ts_event'] = pd.to_datetime(temp_df['ts_event'], unit='ns')
        # Set the datetime column as the DataFrame index for time-series operations
        # and sort the index to ensure chronological order
        temp_df = temp_df.set_index('ts_event').sort_index()
        print(f"Data loaded. Raw date range (Index): {temp_df.index.min()} to {temp_df.index.max()}")

        # --- OHLCV Column Preparation ---
        # Define expected input column names (lowercase) and desired output names (Capitalized)
        ohlcv_cols = {'open': 'Open','high': 'High','low': 'Low','close': 'Close','volume': 'Volume'}
        required_cols = list(ohlcv_cols.keys()) # List of expected input column names

        # Check if all required OHLCV columns exist in the loaded DataFrame
        if not all(col in temp_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in temp_df.columns]
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        # Select only the required OHLCV columns and rename them using the defined mapping
        master_df = temp_df[required_cols].rename(columns=ohlcv_cols)

        # --- Data Type Conversion and Cleaning ---
        # Convert OHLC columns to numeric type (float), coercing errors (non-numeric values become NaN)
        for col in ['Open', 'High', 'Low', 'Close']:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

        # Convert Volume column to numeric (integer), filling potential NaNs with 0 first
        master_df['Volume'] = pd.to_numeric(master_df['Volume'], errors='coerce').fillna(0).astype(int)

        # Check for any NaN values introduced during numeric conversion in OHLC columns
        nan_check_cols = ['Open', 'High', 'Low', 'Close']
        if master_df[nan_check_cols].isnull().any().any():
            nan_rows = master_df[nan_check_cols].isnull().any(axis=1).sum()
            print(f"Warning: {nan_rows} row(s) contain NaN values in OHLC data after conversion. These rows might cause issues or be implicitly dropped later.")
            # Optional: Explicitly drop rows with NaNs in essential columns if desired
            # master_df.dropna(subset=nan_check_cols, inplace=True)
            # print(f"Dropped {nan_rows} rows with NaNs.")

        print("OHLCV data preparation complete.")

    except Exception as e:
        # Catch any error during the loading/preparation process
        print(f"\n--- FATAL ERROR during data loading/preparation ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("Possible causes: Internet connection issue, URL invalid/changed, CSV format error, required columns missing, or data type conversion failure.")
        print("Please check the URL accessibility, file contents, and network connection.")
        print("Script cannot continue without valid data.")
        print("---------------------------------------------------")
        master_df = pd.DataFrame() # Ensure master_df is empty to prevent further steps

    # --- 2. Get User Input ---
    # Initialize variables for user input storage and control flow
    user_timezone = None
    target_day_name = None
    start_date_str, end_date_str = None, None
    range_start_str, range_end_str, post_range_end_str = None, None, None
    analysis_possible = False # Flag to control if analysis can proceed

    # Only attempt to get user input if data loading was successful
    if not master_df.empty:
        try:
            # Call the function to get all user inputs and validate them
            (range_start_str, range_end_str, post_range_end_str,
             target_day_name, target_tz_name,
             start_date_str, end_date_str) = get_user_input()

            # Check if essential inputs (times, timezone) were successfully obtained
            if target_tz_name and range_start_str and range_end_str and post_range_end_str:
                 # Create the ZoneInfo object from the validated timezone name
                 user_timezone = ZoneInfo(target_tz_name)
                 analysis_possible = True # Set flag: We have the minimum inputs to try analysis
            else:
                # This case should ideally be handled within get_user_input, but double-check
                print("\nEssential inputs (Times, Timezone) were not provided or validated. Cannot proceed.")
                analysis_possible = False
        except Exception as e:
             # Catch any unexpected error during input or timezone object creation
             print(f"\nFATAL ERROR during user input processing or timezone creation: {e}")
             analysis_possible = False # Prevent analysis if error occurs here
    else:
        # This message is printed if data loading failed in Step 1
        print("\nSkipping user input: data loading failed.")

    # --- 3. Apply Date Filter ---
    filtered_df = pd.DataFrame() # Initialize empty DataFrame for potentially filtered data

    # Proceed only if data loaded and essential inputs were obtained
    if analysis_possible and not master_df.empty:
        # Work on a copy of the master DataFrame to avoid modifying the original loaded data
        df_to_filter = master_df.copy()

        # Apply date filter only if both start and end dates were provided by the user
        if start_date_str and end_date_str:
            try:
                # Convert string dates provided by user into date objects for comparison
                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                print(f"\nApplying date filter: {start_date_obj} to {end_date_obj} (inclusive)...")

                # Filter the DataFrame index based on the date range
                # Use .index.date to compare only the date part of the DatetimeIndex
                filtered_df = df_to_filter[
                    (df_to_filter.index.date >= start_date_obj) &
                    (df_to_filter.index.date <= end_date_obj)
                ]

                # Check if any data remains after filtering
                if filtered_df.empty:
                    print(f"Warning: No data found within the specified date range {start_date_str} to {end_date_str}.")
                    analysis_possible = False # Stop analysis if filter yields no data
                else:
                    print(f"Data filtered. New date range: {filtered_df.index.min()} to {filtered_df.index.max()}")
            except ValueError as ve:
                # Handle potential errors during date string parsing
                print(f"Error: Invalid date format during filtering ({ve}). Cannot apply date range.")
                analysis_possible = False # Stop analysis
        else:
            # No date filter was chosen by the user
            print("\nNo date filter applied. Using all loaded data.")
            filtered_df = df_to_filter # Use the unfiltered (but prepared) data

        # Final check: even if analysis was possible, did filtering make the df empty?
        if filtered_df.empty and analysis_possible:
            print("Error: Dataframe became empty after potential filtering steps. Cannot proceed with analysis.")
            analysis_possible = False

    # --- 4. Timezone Conversion & Analysis ---
    # Initialize variables to store results from the analysis function
    days_processed = 0; breakout_days = 0
    b_w_pre = 0; post_g_pre = 0; b_no_pre = 0; post_g_no_pre = 0
    # Initialize lists to hold the conditional interval counts
    intervals_pre = [0] * 5; intervals_no_pre = [0] * 5
    analysis_completed = False # Flag to track if the core analysis function ran successfully

    # Proceed only if all previous steps were successful and we have data and a timezone
    if analysis_possible and not filtered_df.empty and user_timezone:
        try:
            print(f"\nApplying timezone localization/conversion to {user_timezone.key}...")

            # --- Timezone Handling ---
            # Ensure the DataFrame index is timezone-aware and set to the target timezone.
            # Assumption: The raw 'ts_event' was in UTC. If not, the localization step needs adjustment.
            if filtered_df.index.tz is None:
                 # If index is naive (no timezone info), assume it's UTC and localize it.
                 print("Index is timezone-naive. Assuming UTC and localizing...")
                 filtered_df.index = filtered_df.index.tz_localize('UTC', ambiguous='infer', nonexistent='NaT')
            else:
                 # If index already has a timezone, convert it to UTC first for a consistent base,
                 print(f"Index has timezone {filtered_df.index.tz}. Converting first to UTC then to target...")
                 filtered_df.index = filtered_df.index.tz_convert('UTC')

            # Convert from UTC (or the localized timezone) to the target user timezone
            filtered_df = filtered_df.tz_convert(user_timezone)
            print(f"Timezone conversion complete. Index is now timezone-aware in {user_timezone.key}.")
            print(f"Data range after TZ conversion: {filtered_df.index.min()} to {filtered_df.index.max()}")

            # --- Call Core Analysis Function ---
            # Pass the prepared DataFrame and parameters to the main analysis logic
            (days_processed, breakout_days,
             b_w_pre, post_g_pre, b_no_pre, post_g_no_pre,
             intervals_pre, intervals_no_pre) = analyze_dynamic_range_rto(
                filtered_df, range_start_str, range_end_str, post_range_end_str, target_day_name, user_timezone
             )
            analysis_completed = True # Mark analysis as successfully run

        except AmbiguousTimeError as ate:
             # Handle errors specifically related to ambiguous times during DST changes
             print(f"\n--- AMBIGUOUS TIME ERROR during Timezone Conversion ---")
             print(f"Error Details: {ate}")
             print(f"Timestamp: {ate.args[0] if ate.args else 'N/A'}")
             print(f"Occurred in target timezone '{user_timezone.key}'. Consider data filtering or different 'ambiguous' handling.")
             print("-----------------------------------------------------------")
             analysis_possible = False # Stop analysis
        except NonExistentTimeError as nete:
            # Handle errors specifically related to times that don't exist during DST changes
            print(f"\n--- NON-EXISTENT TIME ERROR during Timezone Conversion ---")
            print(f"Error Details: {nete}")
            print(f"Timestamp: {nete.args[0] if nete.args else 'N/A'}")
            print(f"Occurred in target timezone '{user_timezone.key}'. Consider data filtering or different 'nonexistent' handling.")
            print("-----------------------------------------------------------")
            analysis_possible = False # Stop analysis
        except Exception as e:
            # Catch any other unexpected errors during timezone conversion or the analysis function call
            print(f"\n--- ERROR during Timezone Conversion or Analysis Execution ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            # import traceback # Uncomment traceback import at top if needed
            # traceback.print_exc() # Uncomment for detailed stack trace
            print("-----------------------------------------------------------")
            analysis_possible = False # Stop analysis
    else:
        # Print message if analysis was skipped due to conditions checked before this block
        if not analysis_possible:
             print("\nAnalysis skipped due to earlier errors or missing input.")


    # --- 5. Display Results (NO DEBUG) ---
    print("\n--- Dynamic Range Breakout & RTO Analysis Results ---")

    if analysis_completed:
        # --- Print Summary of Parameters Used ---
        day_analyzed_desc = target_day_name if target_day_name else "All Applicable Days"
        day_processed_desc = f"'{target_day_name}' days" if target_day_name else "days"
        date_range_desc = f"{start_date_str} to {end_date_str}" if start_date_str and end_date_str else "All Dates in Filtered Data"
        print(f"Date Range Analyzed:      {date_range_desc}")
        print(f"Day(s) Analyzed:          {day_analyzed_desc}")
        print(f"Timezone:                 {user_timezone.key}")
        print(f"Initial Range Window:     {range_start_str} - <{range_end_str} ({user_timezone.key})")
        print(f"RTO Reference Price:      Open at {range_end_str} bar ({user_timezone.key})")
        print(f"Post-Range Window:        {range_end_str} - <{post_range_end_str} ({user_timezone.key})")
        print("-" * 60)

        # --- Print Core Counts ---
        print(f"Total {day_processed_desc} processed (matching filters w/ data in initial range): {days_processed}")
        print(f"Total days with Breakout after Initial Range: {breakout_days}")
        print("-" * 60)

        # --- Overall & Conditional Probability Analysis ---
        # Calculate total post RTO events by summing the conditional counts obtained from analysis
        total_post_rto_events_calc = post_g_pre + post_g_no_pre
        print(f"Total days with Post-Breakout RTO (summed from scenarios): {total_post_rto_events_calc}")

        # Display overall probability of Post-RTO given any breakout occurred
        if breakout_days > 0:
            overall_rto_probability = (total_post_rto_events_calc / breakout_days) * 100
            print(f"Overall Probability of Post-Breakout RTO (given any Breakout): {overall_rto_probability:.2f}%")
        else:
            print("Overall Probability of Post-Breakout RTO (given any Breakout): 0.00% (No breakouts occurred)")
        print("-" * 60)

        # Display conditional probability analysis results
        print("Conditional Post-Breakout RTO Probability Analysis:")
        # Scenario 1: Pre-Breakout RTO Occurred
        print(f"  Days with Breakout AND Pre-Breakout RTO: {b_w_pre}")
        print(f"    (Count where Post-Breakout RTO also happened: {post_g_pre})")
        if b_w_pre > 0:
            # Calculate P(PostRTO | PreRTO) = Days(Pre AND Post) / Days(Pre)
            prob_post_given_pre = (post_g_pre / b_w_pre) * 100
            print(f"  >> Probability of Post-Breakout RTO GIVEN Pre-Breakout RTO occurred: {prob_post_given_pre:.2f}%")
        else:
            # Handle division by zero if no days had a pre-breakout RTO
            print("  >> Probability of Post-Breakout RTO GIVEN Pre-Breakout RTO occurred: N/A (No days with Pre-Breakout RTO)")
        print("-" * 30) # Separator

        # Scenario 2: No Pre-Breakout RTO Occurred
        print(f"  Days with Breakout AND NO Pre-Breakout RTO: {b_no_pre}")
        print(f"    (Count where Post-Breakout RTO also happened: {post_g_no_pre})")
        if b_no_pre > 0:
            # Calculate P(PostRTO | No PreRTO) = Days(No Pre AND Post) / Days(No Pre)
            prob_post_given_no_pre = (post_g_no_pre / b_no_pre) * 100
            print(f"  >> Probability of Post-Breakout RTO GIVEN NO Pre-Breakout RTO occurred: {prob_post_given_no_pre:.2f}%")
        else:
             # Handle division by zero (e.g., if all breakout days had a pre-RTO, or no breakouts)
             print("  >> Probability of Post-Breakout RTO GIVEN NO Pre-Breakout RTO occurred: N/A (No breakout days without Pre-RTO)")
        print("-" * 60)

        # --- Conditional Interval Distribution ---
        print("Conditional Post-Breakout RTO Interval Distribution:")
        print("-" * 60) # Separator line

        # Use the helper function to display results for days WITH Pre-RTO
        display_interval_results(
            title="Interval Distribution GIVEN Pre-Breakout RTO Occurred:",
            interval_counts=intervals_pre,    # Pass the counts for this scenario
            denominator=b_w_pre,              # Denominator is the count of days WITH Pre-RTO
            range_end_str=range_end_str,      # Pass time strings for label calculation
            post_range_end_str=post_range_end_str
        )

        # Use the helper function to display results for days WITHOUT Pre-RTO
        display_interval_results(
            title="Interval Distribution GIVEN NO Pre-Breakout RTO Occurred:",
            interval_counts=intervals_no_pre, # Pass the counts for this scenario
            denominator=b_no_pre,             # Denominator is the count of days with NO Pre-RTO
            range_end_str=range_end_str,      # Pass time strings for label calculation
            post_range_end_str=post_range_end_str
        )
        # --- End of Conditional Interval Distribution ---

    else:
        # Message if analysis did not complete successfully
        print("Analysis could not be completed.")
        print("Please review any error messages printed above in the console.")
        print("Check inputs, data availability/format, network connection, and timezone validity.")

    # --- How to Run Instructions ---
    # 1. Ensure you have Python installed (version 3.9+ recommended for zoneinfo).
    # 2. Install required libraries via pip:
    #    `pip install pandas tzdata`
    # 3. Save this entire code block as a Python file (e.g., `rto_analyzer_final_v2.py`).
    # 4. Open a terminal or command prompt.
    # 5. Navigate to the directory where you saved the file.
    # 6. Run the script using:
    #    `python rto_analyzer_final_v2.py`
    # 7. Follow the prompts.