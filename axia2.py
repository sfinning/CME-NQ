# -*- coding: utf-8 -*-
import pandas as pd
from datetime import time, datetime, timedelta, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import sys
import math # Import math for ceiling function

# --- Function to Analyze Dynamic Range Breakout and Return to Range End Open ---
# Tracks RTO occurrences within 5 sub-intervals of the post-range window
def analyze_dynamic_range_rto(df, range_start_str, range_end_str, post_range_end_str, target_day_name, user_timezone):
    """
    Analyzes price action, tracking RTO events within 5 intervals post-range.
    For each relevant day:
    1. Finds High/Low of initial range.
    2. Gets Open price at range_end_str.
    3. Finds first breakout after range_end_str.
    4. If breakout occurs, finds first RTO after breakout but before post_range_end_str.
    5. If RTO occurs, determines which of 5 equal sub-intervals (of the post-range window)
       it falls into and increments the corresponding counter.

    Returns: (matching_days_processed, breakout_days_count, rto_counts_by_interval)
             where rto_counts_by_interval is a list of 5 integers.
    """
    day_description = target_day_name + "s" if target_day_name else "all days"
    print(f"\nAnalyzing Dynamic Range RTO sequence for {day_description}")
    print(f"Timezone for Analysis: {user_timezone.key}")
    print(f"Initial Range Window: {range_start_str} - <{range_end_str} ({user_timezone.key})")
    print(f"RTO Reference Price: Open at {range_end_str} bar ({user_timezone.key})")
    print(f"Post-Range Window: {range_end_str} - <{post_range_end_str} ({user_timezone.key})")

    matching_days_processed = 0 # Days with data in initial range
    breakout_days_count = 0     # Days with a breakout event after initial range
    rto_counts_by_interval = [0] * 5 # RTO counts for each of the 5 intervals

    # --- Calculate interval durations ---
    try:
        range_start_t = time.fromisoformat(range_start_str)
        range_end_t = time.fromisoformat(range_end_str)
        post_range_end_t = time.fromisoformat(post_range_end_str)
        dummy_date = date.min
        start_dt_dummy = datetime.combine(dummy_date, range_end_t)
        end_dt_dummy = datetime.combine(dummy_date, post_range_end_t)
        total_post_range_duration = end_dt_dummy - start_dt_dummy
        if total_post_range_duration <= timedelta(0):
             print("\nError: Post-Range Window duration must be positive. Cannot proceed.")
             return 0, 0, [0]*5
        interval_duration = total_post_range_duration / 5
        print(f"Post-Range Duration: {total_post_range_duration}, Interval Duration: {interval_duration}")
    except (ValueError, TypeError):
        print("\nError: Invalid time format/inputs. Cannot calculate intervals.")
        return 0, 0, [0]*5

    # --- Filter data ---
    if target_day_name:
        data_to_process = df[df.index.day_name() == target_day_name]
        if data_to_process.empty: return 0, 0, [0]*5
    else:
        data_to_process = df
        if data_to_process.empty: return 0, 0, [0]*5

    unique_days_in_data = data_to_process.index.normalize().unique()
    total_days_available_full = len(unique_days_in_data)

    if total_days_available_full > 0:
         print(f"Found {total_days_available_full} unique day(s) matching criteria. Processing all.")

    # --- Process each day ---
    for day_date in unique_days_in_data:
        day_data = data_to_process[data_to_process.index.date == day_date.date()]
        if day_data.empty: continue

        current_day_date_part = day_date.date()
        try:
            # Define day's specific datetime boundaries
            range_start_dt = datetime.combine(current_day_date_part, range_start_t, tzinfo=user_timezone)
            range_end_dt = datetime.combine(current_day_date_part, range_end_t, tzinfo=user_timezone)
            post_range_end_dt_day = datetime.combine(current_day_date_part, post_range_end_t, tzinfo=user_timezone)
            range_end_exclusive_dt = range_end_dt - timedelta(microseconds=1)
            post_range_end_exclusive_dt_day = post_range_end_dt_day - timedelta(microseconds=1)

            # Calculate interval boundaries for THIS specific day
            interval_boundaries_day = []
            current_boundary_dt = range_end_dt
            for i in range(5):
                end_dt_calc = current_boundary_dt + interval_duration
                next_boundary_dt = min(end_dt_calc, post_range_end_dt_day)
                if i == 4: next_boundary_dt = post_range_end_dt_day # Force last end
                interval_boundaries_day.append((current_boundary_dt, next_boundary_dt))
                current_boundary_dt = next_boundary_dt
        except Exception as e:
            print(f"Warning: Could not create window/interval boundary for {current_day_date_part}. Skipping day. Error: {e}")
            continue

        # --- 1. Analyze Initial Range ---
        initial_range_data = day_data.loc[range_start_dt:range_end_exclusive_dt]
        if initial_range_data.empty: continue
        matching_days_processed += 1
        range_high = initial_range_data['High'].max()
        range_low = initial_range_data['Low'].min()

        # --- 2. Get Ref Price & Post-Range Data ---
        post_range_data = day_data.loc[range_end_dt:post_range_end_exclusive_dt_day]
        if post_range_data.empty: continue
        try: reference_open_price = post_range_data['Open'].iloc[0]
        except IndexError: continue

        # --- 3. Find First Breakout ---
        first_breakout_time = None
        high_break_condition = post_range_data['High'] > range_high
        first_high_break_time = post_range_data[high_break_condition].index.min() if high_break_condition.any() else None
        low_break_condition = post_range_data['Low'] < range_low
        first_low_break_time = post_range_data[low_break_condition].index.min() if low_break_condition.any() else None

        if first_high_break_time and first_low_break_time: first_breakout_time = min(first_high_break_time, first_low_break_time)
        elif first_high_break_time: first_breakout_time = first_high_break_time
        elif first_low_break_time: first_breakout_time = first_low_break_time

        # --- 4. Check for RTO & Determine Interval ---
        if first_breakout_time:
            breakout_days_count += 1
            rto_check_data = post_range_data[post_range_data.index > first_breakout_time]
            if not rto_check_data.empty:
                rto_condition = (rto_check_data['Low'] <= reference_open_price) & (rto_check_data['High'] >= reference_open_price)
                if rto_condition.any():
                    try:
                        first_rto_time = rto_check_data[rto_condition].index.min()
                        for i in range(5):
                            start_interval, end_interval = interval_boundaries_day[i]
                            if first_rto_time >= start_interval and first_rto_time < end_interval:
                                rto_counts_by_interval[i] += 1
                                break
                    except ValueError: pass

    print(f"\nAnalysis complete.")
    if matching_days_processed == 0 and total_days_available_full > 0:
        print(f"No days within the filtered criteria had data in the initial range window {range_start_str}-<{range_end_str} ({user_timezone.key}).")

    return matching_days_processed, breakout_days_count, rto_counts_by_interval


# --- Helper Function for Time Bucket Labels (Unused) ---
def get_bucket_label(bucket_index, interval_minutes=12):
    """Calculates a time bucket label string."""
    start_minute_of_day = bucket_index * interval_minutes
    end_minute_of_day = min(start_minute_of_day + interval_minutes - 1, 1439)
    start_h, start_m = divmod(start_minute_of_day, 60)
    end_h, end_m = divmod(end_minute_of_day, 60)
    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"

# --- Function to get and validate user input ---
# (Remains unchanged)
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
    start_date_str, end_date_str = None, None

    print("--- Enter Analysis Time Windows (Local Time) ---")
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

    print("\n--- Enter Timezone ---")
    while True:
        tz_input = input("Enter Timezone (e.g., EST, CST, UTC, America/New_York, Europe/London): ").strip()
        resolved_tz_name = None; tz_input_upper = tz_input.upper()
        if tz_input_upper in timezone_shortcuts: resolved_tz_name = timezone_shortcuts[tz_input_upper]; print(f"Shortcut mapped to '{resolved_tz_name}'.")
        else: resolved_tz_name = tz_input
        try: ZoneInfo(resolved_tz_name); target_tz_name = resolved_tz_name; print(f"Using timezone: {target_tz_name}"); break
        except ZoneInfoNotFoundError: print(f"Error: Timezone '{resolved_tz_name}' not found.")
        except Exception as e: print(f"Unexpected error validating timezone '{resolved_tz_name}': {e}")

    print("\n--- Optional Day of Week Filter ---")
    while True:
        filter_choice = input("Filter by a specific day of the week? (yes/no): ").strip().lower()
        if filter_choice in ['yes', 'y']:
            while True:
                day_input_str = input(f"Enter Day Number ({allowed_nums_str}): ")
                try: day_num = int(day_input_str); target_day_name = day_number_map[day_num]; print(f"Selected day: {target_day_name}"); break
                except (ValueError, KeyError): print(f"Invalid number. Please enter one of: {allowed_nums_str}.")
            break
        elif filter_choice in ['no', 'n']: target_day_name = None; print("Analyzing all days of the week."); break
        else: print("Invalid input. Please enter 'yes' or 'no'.")

    print("\n--- Optional Date Range Filter ---")
    while True:
        date_filter_choice = input("Filter by a specific date range? (yes/no): ").strip().lower()
        if date_filter_choice in ['yes', 'y']:
            while True:
                s_date_input = input("Enter Analysis Start Date (YYYY-MM-DD format): ")
                try: start_dt_obj = datetime.strptime(s_date_input, '%Y-%m-%d').date(); start_date_str = s_date_input; break
                except ValueError: print("Invalid date format. Please use YYYY-MM-DD.")
            while True:
                e_date_input = input(f"Enter Analysis End Date (YYYY-MM-DD format, on or after {start_date_str}): ")
                try:
                    end_dt_obj = datetime.strptime(e_date_input, '%Y-%m-%d').date()
                    if end_dt_obj >= start_dt_obj: end_date_str = e_date_input; break
                    else: print("End Date must be the same as or after Start Date.")
                except ValueError: print("Invalid date format. Please use YYYY-MM-DD.")
            print(f"Date range selected: {start_date_str} to {end_date_str}"); break
        elif date_filter_choice in ['no', 'n']: start_date_str, end_date_str = None, None; print("Analyzing all dates in the data."); break
        else: print("Invalid input. Please enter 'yes' or 'no'.")

    return range_start_str, range_end_str, post_range_end_str, target_day_name, target_tz_name, start_date_str, end_date_str


# --- Main Script ---
# --- 1. Configuration and Data Loading ---
url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv'
master_df = pd.DataFrame()

# (Data Loading code remains unchanged)
try:
    print("Loading data...")
    temp_df = pd.read_csv(url)
    if 'ts_event' not in temp_df.columns or 'symbol' not in temp_df.columns: raise ValueError("Essential columns 'ts_event' or 'symbol' missing.")
    temp_df['ts_event'] = pd.to_datetime(temp_df['ts_event'], unit='ns')
    temp_df = temp_df.set_index('ts_event').sort_index()
    print(f"Data loaded. Range: {temp_df.index.min()} to {temp_df.index.max()}")
    ohlcv_cols = {'open': 'Open','high': 'High','low': 'Low','close': 'Close','volume': 'Volume'}
    required_cols = list(ohlcv_cols.keys())
    if all(col in temp_df.columns for col in required_cols):
        master_df = temp_df[required_cols].rename(columns=ohlcv_cols)
        for col in master_df.columns:
             if col != 'Volume': master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
        master_df['Volume'] = pd.to_numeric(master_df['Volume'], errors='coerce').fillna(0).astype(int)
        if master_df[['Open', 'High', 'Low', 'Close']].isnull().any().any(): print("Warning: NaN values found in OHLC data.")
        print("OHLCV data preparation complete.")
    else: raise ValueError(f"Missing required OHLCV columns: {[c for c in required_cols if c not in temp_df.columns]}")
except Exception as e: print(f"Error during data loading/preparation: {e}"); master_df = pd.DataFrame()

# --- 2. Get User Input ---
user_timezone = None; target_day_name = None; start_date_str, end_date_str = None, None
range_start_str, range_end_str, post_range_end_str = None, None, None; analysis_possible = False
if not master_df.empty:
    range_start_str, range_end_str, post_range_end_str, target_day_name, target_tz_name, start_date_str, end_date_str = get_user_input()
    if target_tz_name and range_start_str and range_end_str and post_range_end_str:
        try: user_timezone = ZoneInfo(target_tz_name); analysis_possible = True
        except Exception as e: print(f"\nFATAL ERROR creating timezone '{target_tz_name}': {e}")
    else: print("\nEssential inputs missing.")
else: print("\nSkipping user input: data loading failed.")

# --- 3. Apply Date Filter ---
filtered_df = pd.DataFrame()
if analysis_possible:
    df_to_filter = master_df.copy()
    if start_date_str and end_date_str:
        try:
            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date(); end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            print(f"\nApplying date filter: {start_date_obj} to {end_date_obj} (inclusive)...")
            filtered_df = df_to_filter[ (df_to_filter.index.date >= start_date_obj) & (df_to_filter.index.date <= end_date_obj) ]
            if filtered_df.empty: print("Warning: No data in date range."); analysis_possible = False
            else: print(f"Data filtered. Range: {filtered_df.index.min()} to {filtered_df.index.max()}")
        except ValueError: print("Error: Invalid date format during filtering."); analysis_possible = False
    else: print("\nNo date filter applied."); filtered_df = df_to_filter
    if filtered_df.empty and analysis_possible: print("Error: Dataframe empty."); analysis_possible = False # Check again if empty

# --- 4. Timezone Conversion & Analysis ---
days_processed = 0; breakout_days = 0; rto_interval_counts = [0] * 5; analysis_completed = False
if analysis_possible and not filtered_df.empty:
    try:
        print(f"\nLocalizing index to UTC and converting to {user_timezone.key}...")
        filtered_df.index = filtered_df.index.tz_localize('UTC', ambiguous='infer').tz_convert(user_timezone)
        print("Timezone conversion complete.")
        days_processed, breakout_days, rto_interval_counts = analyze_dynamic_range_rto(filtered_df, range_start_str, range_end_str, post_range_end_str, target_day_name, user_timezone)
        analysis_completed = True
    except Exception as e: print(f"\nError during timezone conversion or analysis: {e}")
else: pass # Reason already printed

# --- 5. Display Results ---
print("\n--- Dynamic Range Breakout & RTO Analysis Results ---")
if analysis_completed:
    day_analyzed_desc = target_day_name if target_day_name else "All Days"; day_processed_desc = f"'{target_day_name}' days" if target_day_name else "days"
    date_range_desc = f"{start_date_str} to {end_date_str}" if start_date_str and end_date_str else "All Dates"
    print(f"Date Range Analyzed:      {date_range_desc}"); print(f"Days Analyzed:            {day_analyzed_desc}")
    print(f"Timezone:                 {user_timezone.key}"); print(f"Initial Range Window:     {range_start_str} - <{range_end_str} (Local Time)")
    print(f"RTO Reference Price:      Open at {range_end_str} bar (Local Time)"); print(f"Post-Range Window:        {range_end_str} - <{post_range_end_str} (Local Time)")
    print("-" * 60); print(f"Total {day_processed_desc} processed (matching filters w/ data in initial range): {days_processed}")
    print(f"Total days with Breakout after Initial Range: {breakout_days}"); print("-" * 60)

    # Display Total RTO Probability
    if breakout_days > 0:
        total_rto_events = sum(rto_interval_counts); overall_rto_probability = (total_rto_events / breakout_days) * 100
        print(f"Total days with RTO post-Breakout (within Post-Range Window): {total_rto_events}")
        # Slightly adjusted label:
        print(f"Total Probability of RTO after Breakout: {overall_rto_probability:.2f}%")
    else:
        print("Total days with RTO post-Breakout (within Post-Range Window): 0")
        print("Total Probability of RTO after Breakout: 0.00% (No breakouts occurred)")
    print("-" * 60)

    # Display interval results
    print(f"RTO Occurrences & Probability (given Breakout) per Interval:")
    if breakout_days > 0:
        try:
            range_end_t_obj = time.fromisoformat(range_end_str); post_range_end_t_obj = time.fromisoformat(post_range_end_str)
            start_dt_dummy = datetime.combine(date.min, range_end_t_obj); end_dt_dummy = datetime.combine(date.min, post_range_end_t_obj)
            total_duration = end_dt_dummy - start_dt_dummy; interval_duration = total_duration / 5
            current_interval_start_t = range_end_t_obj
            for i in range(5):
                 current_interval_end_dt = datetime.combine(date.min, current_interval_start_t) + interval_duration; current_interval_end_t = current_interval_end_dt.time()
                 if i == 4: current_interval_end_t = post_range_end_t_obj # Ensure last interval ends exactly
                 start_t_str = current_interval_start_t.strftime('%H:%M:%S'); end_t_str = current_interval_end_t.strftime('%H:%M:%S')
                 interval_label = f"{start_t_str} - <{end_t_str}"; rto_count = rto_interval_counts[i]; probability = (rto_count / breakout_days) * 100
                 print(f"  Interval {i+1} ({interval_label}): Count={rto_count}, Probability={probability:.2f}%")
                 current_interval_start_t = current_interval_end_t # Update start for next loop
        except Exception as e:
            print(f"  Error calculating/displaying interval times: {e}")
            for i in range(5): print(f"  Interval {i+1}: Count={rto_interval_counts[i]}, Probability={(rto_interval_counts[i]/breakout_days)*100:.2f}%")
    else: print("  (No breakouts occurred, cannot calculate interval probabilities)")
    print("-" * 60)
else:
    print("Analysis could not be completed."); print("Check inputs, data availability, and console messages.")

# --- How to Run ---
# 1. Install pandas, tzdata: `pip install pandas tzdata`
# 2. Save as Python file (e.g., `analyzer_intervals_final.py`).
# 3. Run: `python analyzer_intervals_final.py`
# 4. Follow prompts.