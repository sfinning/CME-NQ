# -*- coding: utf-8 -*-
import pandas as pd
from datetime import time, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import sys

# --- Function to Analyze Daily Breakouts, RTO, and Pre-RTO Trade Window Extreme Times ---
# (analyze_daily_breakouts_rto_times function remains unchanged from the previous version)
def analyze_daily_breakouts_rto_times(df, obs_start_str, obs_end_str, target_day_name, trade_minutes, user_timezone, day_limit=None):
    """
    Analyzes breakouts, RTO (Return to Trade Window's first bar Open, occurring
    after the first bar), and the time of the TRADE window extreme that occurred
    AFTER the first breakout BUT BEFORE the first RTO event, on days where
    breakout occurred chronologically BEFORE RTO.
    Observation window is exclusive of the end time.
    Trade window is inclusive of the start time.
    Assumes df.index is already localized to user_timezone.

    Returns: (matching_days_processed, bullish_bo_count, bearish_bo_count,
              rto_after_bullish_count, rto_after_bearish_count,
              rto_bullish_extreme_times, rto_bearish_extreme_times)
    """
    print(f"\nAnalyzing breakouts, RTO (Return to TW Open, after T1; BO before RTO), and Pre-RTO timing for {target_day_name}s")
    print(f"Timezone for Input Times: {user_timezone.key}")
    print(f"Observation Window: {obs_start_str} - <{obs_end_str} (Exclusive, {user_timezone.key})")
    print(f"Trade Window: {obs_end_str} + {trade_minutes} mins (Inclusive Start, {user_timezone.key})")

    matching_days_processed = 0
    bullish_breakout_days_count = 0
    bearish_breakout_days_count = 0
    rto_after_bullish_breakout_count = 0
    rto_after_bearish_breakout_count = 0
    rto_bullish_extreme_times = []
    rto_bearish_extreme_times = []

    try:
        obs_start_t = time.fromisoformat(obs_start_str)
        obs_end_t = time.fromisoformat(obs_end_str)
    except ValueError:
        print("Error: Invalid time format provided for Observation Window.")
        return 0, 0, 0, 0, 0, [], []

    data_for_day = df[df.index.day_name() == target_day_name]
    if data_for_day.empty:
        print(f"No data found for any {target_day_name} in the dataset (Timezone: {user_timezone.key}).")
        return 0, 0, 0, 0, 0, [], []

    unique_days_in_data = data_for_day.index.normalize().unique()
    total_days_available_full = len(unique_days_in_data)
    unique_days_to_process = unique_days_in_data
    if day_limit is not None and day_limit > 0:
        print(f"Found {total_days_available_full} unique {target_day_name}s.")
        if total_days_available_full > day_limit:
            print(f"Limiting analysis to the first {day_limit} day(s).")
            unique_days_to_process = unique_days_in_data[:day_limit]
        else: print(f"Processing all {total_days_available_full} found day(s).")
    else: print(f"Found {total_days_available_full} unique {target_day_name}s. Processing all.")

    for day_date in unique_days_to_process:
        day_data = data_for_day[data_for_day.index.date == day_date.date()]
        if day_data.empty: continue
        current_day_date_part = day_date.date()
        try:
            obs_start_dt = datetime.combine(current_day_date_part, obs_start_t, tzinfo=user_timezone)
            obs_end_dt = datetime.combine(current_day_date_part, obs_end_t, tzinfo=user_timezone)
        except Exception as e:
            print(f"Warning: Could not create window boundary for {current_day_date_part} at {obs_start_t}/{obs_end_t} in {user_timezone.key}. Skipping day. Error: {e}")
            continue

        obs_end_exclusive_dt = obs_end_dt - timedelta(microseconds=1)
        trade_start_dt = obs_end_dt
        trade_end_dt = trade_start_dt + timedelta(minutes=trade_minutes)
        day_end_dt = datetime.combine(current_day_date_part, time(23, 59, 59, 999999), tzinfo=user_timezone)
        actual_trade_end_dt = min(trade_end_dt, day_end_dt)
        day_obs_data = day_data.loc[obs_start_dt:obs_end_exclusive_dt]
        if day_obs_data.empty: continue
        matching_days_processed += 1
        day_obs_high_val = day_obs_data['High'].max()
        day_obs_low_val = day_obs_data['Low'].min()
        trade_window_data_day = day_data.loc[trade_start_dt:actual_trade_end_dt]

        high_broken, low_broken = False, False
        first_high_break_timestamp, first_low_break_timestamp = None, None
        if not trade_window_data_day.empty:
            high_break_condition = trade_window_data_day['High'] > day_obs_high_val
            low_break_condition = trade_window_data_day['Low'] < day_obs_low_val
            high_broken, low_broken = high_break_condition.any(), low_break_condition.any()
            if high_broken:
                try: first_high_break_timestamp = trade_window_data_day[high_break_condition].index.min()
                except ValueError: high_broken = False; print(f"Warning: Could not determine first high break time for {day_date.date()} despite high_broken=True.")
            if low_broken:
                try: first_low_break_timestamp = trade_window_data_day[low_break_condition].index.min()
                except ValueError: low_broken = False; print(f"Warning: Could not determine first low break time for {day_date.date()} despite low_broken=True.")

        rto_occurred_later, first_rto_timestamp, trade_window_open_price = False, None, None
        if not trade_window_data_day.empty:
            try:
                trade_window_open_price = trade_window_data_day['Open'].iloc[0]
                trade_window_later_bars = trade_window_data_day.iloc[1:]
                if not trade_window_later_bars.empty and trade_window_open_price is not None:
                    rto_condition_later = (trade_window_later_bars['Low'] <= trade_window_open_price) & (trade_window_later_bars['High'] >= trade_window_open_price)
                    rto_occurred_later = rto_condition_later.any()
                    if rto_occurred_later:
                        try: first_rto_timestamp = trade_window_later_bars[rto_condition_later].index.min()
                        except ValueError: rto_occurred_later = False; print(f"Warning: Could not determine first RTO time (after T1) for {day_date.date()} despite rto_occurred_later=True.")
            except IndexError: trade_window_open_price, rto_occurred_later, first_rto_timestamp = None, False, None; print(f"Warning: Could not get trade window open price for {day_date.date()}.")

        if high_broken:
            bullish_breakout_days_count += 1
            if rto_occurred_later and first_rto_timestamp and first_high_break_timestamp and first_high_break_timestamp < first_rto_timestamp:
                rto_after_bullish_breakout_count += 1
                pre_rto_data = trade_window_data_day[(trade_window_data_day.index >= first_high_break_timestamp) & (trade_window_data_day.index < first_rto_timestamp)]
                if not pre_rto_data.empty:
                    try: rto_bullish_extreme_times.append(pre_rto_data['High'].idxmax())
                    except ValueError: print(f"Warning: Could not find pre-RTO high time on {day_date.date()} for Bullish Breakout -> RTO.")
        if low_broken:
            bearish_breakout_days_count += 1
            if rto_occurred_later and first_rto_timestamp and first_low_break_timestamp and first_low_break_timestamp < first_rto_timestamp:
                rto_after_bearish_breakout_count += 1
                pre_rto_data = trade_window_data_day[(trade_window_data_day.index >= first_low_break_timestamp) & (trade_window_data_day.index < first_rto_timestamp)]
                if not pre_rto_data.empty:
                    try: rto_bearish_extreme_times.append(pre_rto_data['Low'].idxmin())
                    except ValueError: print(f"Warning: Could not find pre-RTO low time on {day_date.date()} for Bearish Breakout -> RTO.")

    print(f"\nAnalysis complete.")
    if matching_days_processed == 0:
        limit_msg = f" among the first {day_limit} checked" if day_limit else ""
        print(f"No {target_day_name}s found with data within the observation window {obs_start_str}-<{obs_end_str} ({user_timezone.key}){limit_msg}.")
    return (matching_days_processed, bullish_breakout_days_count, bearish_breakout_days_count, rto_after_bullish_breakout_count, rto_after_bearish_breakout_count, rto_bullish_extreme_times, rto_bearish_extreme_times)


# --- Helper Function for Time Bucket Labels (Unchanged) ---
def get_bucket_label(bucket_index, interval_minutes=12):
    start_minute_of_day = bucket_index * interval_minutes
    end_minute_of_day = min(start_minute_of_day + interval_minutes - 1, 1439)
    start_h, start_m = divmod(start_minute_of_day, 60)
    end_h, end_m = divmod(end_minute_of_day, 60)
    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"


# --- Function to get and validate user input (MODIFIED) ---
def get_user_input():
    """
    Prompts user for start time, end time, day of week (as number 2-6),
    trade window minutes, and timezone (allowing shortcuts EST, CST, UTC),
    with validation.
    """
    day_number_map = { 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday" }
    allowed_nums_str = ", ".join(map(str, day_number_map.keys()))

    # *** ADDED: Timezone shortcut mapping ***
    timezone_shortcuts = {
        "EST": "America/New_York", # Eastern Time Zone (handles DST)
        "CST": "America/Chicago",  # Central Time Zone (handles DST)
        "UTC": "UTC"               # Coordinated Universal Time
    }

    start_time_str, end_time_str, target_day_name, trade_window_minutes, target_tz_name = None, None, None, None, None

    # --- Get Observation Window ---
    print("--- Enter Observation Window Details ---")
    while True:
        start_input = input("Enter the Observation Start Time (HH:MM format, e.g., 08:30): ")
        try: time.fromisoformat(start_input); start_time_str = start_input; break
        except ValueError: print("Invalid time format. Please use HH:MM (24-hour).")
    while True:
        end_input = input("Enter the Observation End Time (HH:MM format, e.g., 09:00): ")
        try:
            if time.fromisoformat(end_input) > time.fromisoformat(start_time_str): end_time_str = end_input; break
            else: print("End Time must be after Start Time.")
        except ValueError: print("Invalid time format. Please use HH:MM (24-hour).")

    # --- Get Day of Week by Number ---
    print("\n--- Enter Day of Week ---")
    while True:
        day_input_str = input(f"Enter the Day of the Week ({allowed_nums_str}): ")
        try:
            day_num = int(day_input_str)
            if day_num in day_number_map: target_day_name = day_number_map[day_num]; print(f"Selected day: {target_day_name}"); break
            else: print(f"Invalid number. Please enter one of: {allowed_nums_str}")
        except ValueError: print(f"Invalid input. Please enter a number ({allowed_nums_str}).")

    # --- Get Trade Window ---
    print("\n--- Enter Trade Window Details ---")
    while True:
        try:
            minutes_input = int(input("Enter the Trade Window duration in minutes (e.g., 60): "))
            if minutes_input > 0: trade_window_minutes = minutes_input; break
            else: print("Please enter a positive number of minutes.")
        except ValueError: print("Invalid input. Please enter a whole number for minutes.")

    # *** MODIFIED: Get Timezone with shortcuts ***
    print("\n--- Enter Timezone ---")
    print("Enter Olson timezone name or shortcut (EST=America/New_York, CST=America/Chicago, UTC=UTC).")
    print("Examples: EST, CST, UTC, Europe/London, America/Denver")
    while True:
        tz_input = input("Enter Timezone: ").strip()
        resolved_tz_name = None # Variable to hold the name to be validated

        # Check for shortcuts (case-insensitive)
        tz_input_upper = tz_input.upper()
        if tz_input_upper in timezone_shortcuts:
            resolved_tz_name = timezone_shortcuts[tz_input_upper]
            print(f"Shortcut '{tz_input}' mapped to '{resolved_tz_name}'.")
        else:
            # If not a shortcut, assume it's a full Olson name
            resolved_tz_name = tz_input

        # Try to validate the resolved name by creating a ZoneInfo object
        try:
            ZoneInfo(resolved_tz_name)
            target_tz_name = resolved_tz_name # Store the valid Olson name
            print(f"Using timezone: {target_tz_name}")
            break # Exit loop on success
        except ZoneInfoNotFoundError:
            print(f"Error: Timezone '{resolved_tz_name}' not found. Please use a valid Olson name (e.g., 'America/New_York') or shortcut (EST, CST, UTC).")
        except Exception as e:
            print(f"An unexpected error occurred validating timezone '{resolved_tz_name}': {e}")

    # Return all collected inputs (including the resolved Olson timezone name)
    return start_time_str, end_time_str, target_day_name, trade_window_minutes, target_tz_name


# --- Main Script ---
# --- 1. Configuration and Data Loading ---
url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv'
df_analysis_all_symbols = pd.DataFrame()
DAY_LIMIT_FOR_TESTING = 250

try:
    print("Loading data...")
    df = pd.read_csv(url)
    if 'ts_event' not in df.columns or 'symbol' not in df.columns: raise ValueError("Essential columns missing.")
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')
    df = df.set_index('ts_event').sort_index()
    print(f"Data loaded. Index is initially timezone-naive.")
    ohlcv_cols = {'open': 'Open','high': 'High','low': 'Low','close': 'Close','volume': 'Volume'}
    if all(col in df.columns for col in ohlcv_cols.keys()):
        df_analysis_all_symbols = df[list(ohlcv_cols.keys())].rename(columns=ohlcv_cols)
        for col in ohlcv_cols.values(): df_analysis_all_symbols[col] = pd.to_numeric(df_analysis_all_symbols[col], errors='coerce')
        if df_analysis_all_symbols.isnull().sum().sum() > 0: print("Warning: Some non-numeric OHLCV data converted to NaN.")
        print("OHLCV data preparation complete.")
    else: raise ValueError(f"Missing required columns: {[c for c in ohlcv_cols.keys() if c not in df.columns]}")
except Exception as e:
    print(f"An error occurred during data loading/preparation: {e}")
    df_analysis_all_symbols = pd.DataFrame()

# --- 2. Get User Input ---
user_timezone = None
if not df_analysis_all_symbols.empty:
    start_time_str, end_time_str, target_day_name, trade_window_minutes, target_tz_name = get_user_input()
    if target_tz_name:
        try: user_timezone = ZoneInfo(target_tz_name)
        except Exception as e: print(f"\nFATAL ERROR creating timezone '{target_tz_name}': {e}"); user_timezone = None; target_day_name = None; sys.exit(1)
else:
    start_time_str, end_time_str, target_day_name, trade_window_minutes, target_tz_name = (None, None, None, None, None); print("\nSkipping user input due to data loading issues.")

# --- 3. Timezone Conversion & Analysis ---
analysis_df = pd.DataFrame()
if not df_analysis_all_symbols.empty and user_timezone and target_day_name:
    try:
        print(f"\nLocalizing index to UTC and converting to {user_timezone.key}...")
        analysis_df = df_analysis_all_symbols.copy()
        # Localize the naive index to UTC first, then convert
        analysis_df.index = analysis_df.index.tz_localize('UTC', ambiguous='infer').tz_convert(user_timezone)
        print("Timezone conversion complete.")
        (days_processed, bullish_bo_days, bearish_bo_days, rto_bullish_days, rto_bearish_days, bullish_ext_times, bearish_ext_times) = analyze_daily_breakouts_rto_times(
            analysis_df, start_time_str, end_time_str, target_day_name, trade_window_minutes, user_timezone, day_limit=DAY_LIMIT_FOR_TESTING
        )
    except Exception as e:
         print(f"\nAn error occurred during timezone conversion or analysis: {e}")
         target_day_name = None
else:
     target_day_name = None
     if df_analysis_all_symbols.empty: pass
     elif not user_timezone: print("\nAnalysis skipped due to missing or invalid timezone.")
     elif not target_day_name: print("\nAnalysis skipped due to missing day selection.")


# --- 4. Display Results ---
if target_day_name and user_timezone:
    print("\n--- Breakout, RTO (to TW Open, after T1; BO before RTO) & Timing Analysis Results ---")
    if DAY_LIMIT_FOR_TESTING is not None and days_processed > 0 : print(f"--- NOTE: (Analysis limited to first {DAY_LIMIT_FOR_TESTING} {target_day_name}s found) ---")
    print(f"Selected Day:           {target_day_name}")
    print(f"Timezone:               {user_timezone.key}")
    print(f"Observation Window:     {start_time_str} - <{end_time_str} (Exclusive, Local Time)")
    print(f"Trade Window:           {end_time_str} + {trade_window_minutes} mins (Inclusive Start, Local Time)")
    print("-" * 60)
    if days_processed > 0:
        print(f"Total '{target_day_name}' days processed with data in Obs. Window: {days_processed}")
        print("-" * 60)
        # Bullish
        print(f"Days with Bullish Breakout (High Broken): {bullish_bo_days}")
        if bullish_bo_days > 0:
            print(f"  Breakout Percentage (Bullish): {(bullish_bo_days / days_processed) * 100:.2f}%")
            print(f"  RTO Count (Breakout -> RTO to T1 Open occurring >= T2): {rto_bullish_days}")
            print(f"  Probability(RTO>=T2 & BO First | Bullish Breakout): {(rto_bullish_days / bullish_bo_days) * 100 if bullish_bo_days > 0 else 0:.2f}%")
            if bullish_ext_times:
                print("\n  -- Timing of High (Between Breakout and RTO>=T2) --")
                time_series = pd.Series(bullish_ext_times)
                bucket_interval = 12
                bucket_indices = (time_series.dt.hour * 60 + time_series.dt.minute) // bucket_interval
                bucket_counts = bucket_indices.value_counts().sort_index()
                print(f"  Counts per {bucket_interval}-minute interval (Local Time):")
                for idx, count in bucket_counts.items(): print(f"    {get_bucket_label(idx, bucket_interval)}: {count}")
            elif rto_bullish_days > 0: print("\n  -- Warning: Bullish Breakout->RTO(>=T2) occurred but failed to record valid pre-RTO high times --")
            else: print("\n  -- No Bullish Breakout -> RTO(>=T2) sequences found to analyze timing --")
        else: print("  (No bullish breakouts occurred)")
        print("-" * 60)
        # Bearish
        print(f"Days with Bearish Breakout (Low Broken): {bearish_bo_days}")
        if bearish_bo_days > 0:
            print(f"  Breakout Percentage (Bearish): {(bearish_bo_days / days_processed) * 100:.2f}%")
            print(f"  RTO Count (Breakout -> RTO to T1 Open occurring >= T2): {rto_bearish_days}")
            print(f"  Probability(RTO>=T2 & BO First | Bearish Breakout): {(rto_bearish_days / bearish_bo_days) * 100 if bearish_bo_days > 0 else 0:.2f}%")
            if bearish_ext_times:
                print("\n  -- Timing of Low (Between Breakout and RTO>=T2) --")
                time_series = pd.Series(bearish_ext_times)
                bucket_interval = 12
                bucket_indices = (time_series.dt.hour * 60 + time_series.dt.minute) // bucket_interval
                bucket_counts = bucket_indices.value_counts().sort_index()
                print(f"  Counts per {bucket_interval}-minute interval (Local Time):")
                for idx, count in bucket_counts.items(): print(f"    {get_bucket_label(idx, bucket_interval)}: {count}")
            elif rto_bearish_days > 0: print("\n  -- Warning: Bearish Breakout->RTO(>=T2) occurred but failed to record valid pre-RTO low times --")
            else: print("\n  -- No Bearish Breakout -> RTO(>=T2) sequences found to analyze timing --")
        else: print("  (No bearish breakouts occurred)")
        print("-" * 60)
    elif not analysis_df.empty: print(f"No matching days found with data in the observation window to analyze for breakouts.")

# --- How to Run ---
# (Instructions remain the same)