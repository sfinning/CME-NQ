# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
# Import necessary components from datetime, including the specific error types
from datetime import time, datetime, timedelta, date
# Import necessary components from zoneinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import sys
import math # Keep math import (though not used currently)
from pytz import AmbiguousTimeError, NonExistentTimeError # Specific errors for timezone handling

# =============================================================================
# Core Analysis Function
# =============================================================================
# MODIFIED: Added initial_range_type_filter parameter
def analyze_dynamic_range_rto(df, range_start_str, range_end_str, post_range_end_str, target_day_name, user_timezone, initial_range_type_filter="All"):
    """
    Analyzes price action, tracking RTO events.
    MODIFIED: Tracks Post-Breakout RTO interval occurrences separately based on
    whether a Pre-Breakout RTO occurred on the same day.
    MODIFIED: Post-Breakout RTO search now starts strictly AFTER the breakout bar.
    FIXED: Check for empty unique_days_in_data using .empty attribute.
    NEW: Filters days based on whether the initial range was Bullish or Bearish.

    Returns: (matching_days_processed, breakout_days_count,
              breakout_with_pre_rto_count, post_rto_given_pre_rto_count,
              breakout_no_pre_rto_count, post_rto_given_no_pre_rto_count,
              rto_intervals_given_pre_rto, rto_intervals_given_no_pre_rto)
    """
    # --- Initial setup ---
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
        # Use fromisoformat which handles HH:MM and HH:MM:SS
        range_start_t = time.fromisoformat(range_start_str)
        range_end_t = time.fromisoformat(range_end_str)
        post_range_end_t = time.fromisoformat(post_range_end_str)
        dummy_date = date.min
        start_dt_dummy = datetime.combine(dummy_date, range_end_t)
        end_dt_dummy = datetime.combine(dummy_date, post_range_end_t)
        total_post_range_duration = end_dt_dummy - start_dt_dummy
        if total_post_range_duration <= timedelta(0):
             st.warning("Post-range duration is zero or negative. Check time inputs.")
             return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5
        interval_duration = total_post_range_duration / 5
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid time format/inputs for interval calculation: {e}")

    # --- Filter data by day name if specified ---
    if target_day_name:
        data_to_process = df[df.index.day_name() == target_day_name]
        if data_to_process.empty:
            st.warning(f"No data found for the selected day: {target_day_name}")
            return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5
    else:
        data_to_process = df
        if data_to_process.empty:
             raise ValueError("Input DataFrame to analyze_dynamic_range_rto is empty.")

    # Get unique days present in the data to iterate over
    unique_days_in_data = data_to_process.index.normalize().unique()
    total_days_available_full = len(unique_days_in_data)

    if unique_days_in_data.empty:
        st.warning("No unique days found in the filtered data to process.")
        return 0, 0, 0, 0, 0, 0, [0]*5, [0]*5

    # --- Process each unique day ---
    for day_date in unique_days_in_data:
        day_data = data_to_process[data_to_process.index.date == day_date.date()]
        if day_data.empty: continue

        current_day_date_part = day_date.date()
        interval_boundaries_day = []
        try:
            # Combine date with time and add timezone info
            range_start_dt = datetime.combine(current_day_date_part, range_start_t, tzinfo=user_timezone)
            range_end_dt = datetime.combine(current_day_date_part, range_end_t, tzinfo=user_timezone)
            post_range_end_dt_day = datetime.combine(current_day_date_part, post_range_end_t, tzinfo=user_timezone)
            # Define exclusive end times for slicing
            range_end_exclusive_dt = range_end_dt - timedelta(microseconds=1)
            post_range_end_exclusive_dt_day = post_range_end_dt_day - timedelta(microseconds=1)

            # Calculate interval boundaries for this specific day
            current_boundary_dt = range_end_dt
            for i in range(5):
                end_dt_calc = current_boundary_dt + interval_duration
                # Ensure the calculated end doesn't exceed the overall post-range end
                next_boundary_dt = min(end_dt_calc, post_range_end_dt_day)
                # Ensure the very last interval ends exactly at the post_range_end_dt_day
                if i == 4: next_boundary_dt = post_range_end_dt_day
                interval_boundaries_day.append((current_boundary_dt, next_boundary_dt))
                current_boundary_dt = next_boundary_dt

        except Exception as e:
            # Log or warn about skipping day due to potential timezone issues (DST etc.) on combine
            # print(f"Skipping day {current_day_date_part} due to time combination/calculation error: {e}") # Optional: for debugging
            continue # Skip this day if combine fails

        # --- 1. Analyze Initial Range ---
        # Select data strictly within the initial range start and end times
        initial_range_data = day_data.loc[range_start_dt:range_end_exclusive_dt]
        if initial_range_data.empty: continue # Skip day if no data in initial range timeframe

        # ===> NEW: Initial Range Type Filter Logic <===
        if initial_range_type_filter != "All":
            try:
                # Get the open of the very first bar and close of the very last bar in the range
                initial_open = initial_range_data['Open'].iloc[0]
                initial_close = initial_range_data['Close'].iloc[-1]
                is_bullish = initial_close > initial_open
                is_bearish = initial_close < initial_open

                # Apply the filter - skip the day if it doesn't match
                if initial_range_type_filter == "Bullish" and not is_bullish:
                    continue
                if initial_range_type_filter == "Bearish" and not is_bearish:
                    continue
            except IndexError:
                # This might happen if somehow initial_range_data is not truly empty but lacks rows
                continue # Skip if range data is malformed
        # ===> END NEW FILTER LOGIC <===

        # If the day passed the filter (or filter is 'All'), increment processed count
        matching_days_processed += 1
        # Calculate range high/low from the initial range data
        range_high = initial_range_data['High'].max()
        range_low = initial_range_data['Low'].min()

        # --- 2. Get Reference Price & Post-Range Data ---
        # Select data for the post-range analysis window
        post_range_data = day_data.loc[range_end_dt:post_range_end_exclusive_dt_day]
        if post_range_data.empty: continue # Skip if no data in post-range window
        try:
            # The reference price is the Open of the *first* bar at or after range_end_dt
            reference_open_price = post_range_data['Open'].iloc[0]
        except IndexError:
            continue # Skip if post_range_data somehow became empty or has no 'Open'

        # --- 3. Find First Breakout ---
        # Determine the exact time of the first breakout above range high or below range low
        first_breakout_time = None
        high_break_condition = post_range_data['High'] > range_high
        first_high_break_time = post_range_data[high_break_condition].index.min() if high_break_condition.any() else None
        low_break_condition = post_range_data['Low'] < range_low
        first_low_break_time = post_range_data[low_break_condition].index.min() if low_break_condition.any() else None

        # Find the earlier of the two breakout times, if they exist
        if first_high_break_time and first_low_break_time:
            first_breakout_time = min(first_high_break_time, first_low_break_time)
        elif first_high_break_time:
            first_breakout_time = first_high_break_time
        elif first_low_break_time:
            first_breakout_time = first_low_break_time
        # If no breakout occurred (first_breakout_time is None), skip RTO checks for this day

        # --- 4. Check for Pre-RTO, Post-RTO & Determine Interval ---
        if first_breakout_time:
            breakout_days_count += 1 # Increment count of days that had a breakout
            pre_rto_occurred = False
            post_rto_occurred = False
            first_post_rto_time = None

            # --- 4a. Check for Pre-Breakout RTO ---
            # Look for RTO in the bars between range end and the breakout bar
            pre_breakout_check_data = post_range_data[post_range_data.index < first_breakout_time]
            if not pre_breakout_check_data.empty:
                # Check if any bar's High/Low range touched the reference open price
                pre_rto_condition = (pre_breakout_check_data['Low'] <= reference_open_price) & (pre_breakout_check_data['High'] >= reference_open_price)
                if pre_rto_condition.any():
                    pre_rto_occurred = True
                    breakout_with_pre_rto_count += 1
                else:
                    breakout_no_pre_rto_count += 1
            else:
                # If no bars between range end and breakout, it counts as no pre-RTO
                breakout_no_pre_rto_count += 1

            # --- 4b. Check for Post-Breakout RTO (Strictly AFTER breakout bar) ---
            # Look for RTO in bars strictly after the breakout bar's time
            rto_check_data = post_range_data[post_range_data.index > first_breakout_time]
            if not rto_check_data.empty:
                post_rto_condition = (rto_check_data['Low'] <= reference_open_price) & (rto_check_data['High'] >= reference_open_price)
                if post_rto_condition.any():
                     try:
                         # Find the timestamp of the first bar that satisfies the condition
                         first_post_rto_time = rto_check_data[post_rto_condition].index.min()
                         post_rto_occurred = True
                     except ValueError: pass # Should not happen if .any() is true

            # --- 4c. Increment Conditional Counters & Interval Counts ---
            if post_rto_occurred:
                # Increment counts based on whether a pre-RTO also happened
                if pre_rto_occurred:
                    post_rto_given_pre_rto_count += 1
                else:
                    post_rto_given_no_pre_rto_count += 1

                # Determine which interval the first post-RTO occurred in
                if first_post_rto_time:
                    for i in range(5):
                        start_interval, end_interval = interval_boundaries_day[i]
                        is_in_interval = False
                        # Check if RTO time falls within the interval boundaries
                        # Use <= for the end boundary only on the last interval (i=4)
                        if i == 4: # Last interval includes the end boundary
                             if start_interval <= first_post_rto_time <= end_interval:
                                 is_in_interval = True
                        else: # Other intervals are [start, end)
                             if start_interval <= first_post_rto_time < end_interval:
                                 is_in_interval = True

                        if is_in_interval:
                            # Increment the correct interval list based on pre-RTO status
                            if pre_rto_occurred:
                                rto_intervals_given_pre_rto[i] += 1
                            else:
                                rto_intervals_given_no_pre_rto[i] += 1
                            break # RTO can only be in one interval, stop checking
        # --- End of processing for a single day ---

    # Return all calculated counts
    return (matching_days_processed, breakout_days_count,
            breakout_with_pre_rto_count, post_rto_given_pre_rto_count,
            breakout_no_pre_rto_count, post_rto_given_no_pre_rto_count,
            rto_intervals_given_pre_rto, rto_intervals_given_no_pre_rto)

# =============================================================================
# Data Loading Function with Streamlit Caching
# =============================================================================
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(url):
    """Loads data from URL, prepares OHLCV columns, and returns DataFrame."""
    try:
        st.write(f"Loading data from {url}...") # Show status in app
        temp_df = pd.read_csv(url)

        # Basic Column Validation
        if 'ts_event' not in temp_df.columns or 'symbol' not in temp_df.columns:
            raise ValueError("Essential columns 'ts_event' or 'symbol' missing.")

        # Timestamp Conversion and Indexing (assuming nanoseconds UTC)
        temp_df['ts_event'] = pd.to_datetime(temp_df['ts_event'], unit='ns', errors='coerce')
        temp_df.dropna(subset=['ts_event'], inplace=True) # Drop rows where conversion failed
        temp_df = temp_df.set_index('ts_event').sort_index()

        # OHLCV Column Preparation
        ohlcv_cols = {'open': 'Open','high': 'High','low': 'Low','close': 'Close','volume': 'Volume'}
        required_cols = list(ohlcv_cols.keys())

        if not all(col in temp_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in temp_df.columns]
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        master_df = temp_df[required_cols].rename(columns=ohlcv_cols)

        # Data Type Conversion and Cleaning
        for col in ['Open', 'High', 'Low', 'Close']:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
        master_df['Volume'] = pd.to_numeric(master_df['Volume'], errors='coerce').fillna(0).astype(int)

        # Handle potential NaNs from conversion
        nan_check_cols = ['Open', 'High', 'Low', 'Close']
        if master_df[nan_check_cols].isnull().any().any():
            nan_rows = master_df[nan_check_cols].isnull().any(axis=1).sum()
            master_df.dropna(subset=nan_check_cols, inplace=True)
            st.warning(f"Warning: Dropped {nan_rows} row(s) containing NaN values in OHLC data after conversion.")

        if master_df.empty:
             st.error("Data loaded, but became empty after cleaning (NaN removal or timestamp issues).")
             return None # Return None to indicate failure

        st.success("Data loading and preparation complete.") # Success message
        return master_df

    except FileNotFoundError:
        raise ConnectionError(f"Error: Could not find file at URL: {url}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: The file at {url} is empty.")
    except Exception as e:
        # Raise other exceptions to be caught by the main app logic
        raise ConnectionError(f"Error during data loading/preparation: {e}")


# =============================================================================
# Helper Function to Calculate Interval Labels
# =============================================================================
def get_interval_labels(range_end_t, post_range_end_t):
    """Calculates H:M:S labels for the 5 intervals."""
    labels = []
    try:
        dummy_date = date.min
        start_dt_dummy = datetime.combine(dummy_date, range_end_t)
        end_dt_dummy = datetime.combine(dummy_date, post_range_end_t)
        total_duration = end_dt_dummy - start_dt_dummy

        if total_duration <= timedelta(0): raise ValueError("Post-range duration must be positive")

        interval_duration = total_duration / 5
        current_interval_start_dt = start_dt_dummy

        for i in range(5):
            current_interval_end_dt = current_interval_start_dt + interval_duration
            # Ensure the last interval ends exactly at the post_range_end_time
            if i == 4: current_interval_end_dt = end_dt_dummy

            # Format times as HH:MM:SS
            start_t_str = current_interval_start_dt.strftime('%H:%M:%S')
            end_t_str = current_interval_end_dt.strftime('%H:%M:%S')

            # Use '<=' for the end time only for the last interval
            interval_label = f"{start_t_str} - <{end_t_str}"
            if i == 4: interval_label = f"{start_t_str} - <={end_t_str}"
            labels.append(interval_label)
            # The start of the next interval is the end of the current one
            current_interval_start_dt = current_interval_end_dt
    except Exception as e:
        st.warning(f"Could not calculate interval labels: {e}")
        labels = [f"Interval {i+1}" for i in range(5)] # Fallback labels
    return labels


# =============================================================================
# Streamlit Application UI and Logic
# =============================================================================

st.set_page_config(layout="wide") # Use wider layout
st.title("📈 Dynamic Range Breakout & RTO Analyzer")
st.markdown("Analyze the probability and timing of price returning to the initial range end open price after a breakout, with optional filters.")

# --- Sidebar for Inputs ---
st.sidebar.header("Analysis Parameters")

# Time Windows
default_start_time = time(9, 0)
default_end_time = time(10, 0)
default_post_end_time = time(11, 0)
# Use step=60 for minute precision input, help text added
range_start_t = st.sidebar.time_input("Initial Range Start Time", value=default_start_time, help="Start of the initial balance range (inclusive).", step=60)
range_end_t = st.sidebar.time_input("Initial Range End Time", value=default_end_time, help="End of the initial balance range. Reference price is Open of this bar.", step=60)
post_range_end_t = st.sidebar.time_input("Post-Range End Time", value=default_post_end_time, help="End of the window to check for breakouts and RTOs.", step=60)

# Timezone Input
tz_shortcuts = {"EST": "America/New_York", "CST": "America/Chicago", "PST": "America/Los_Angeles", "UTC": "UTC"}
tz_options = list(tz_shortcuts.keys()) + ["Custom"]
selected_tz_option = st.sidebar.selectbox("Select Timezone", options=tz_options, index=tz_options.index("EST"), help="Timezone for defining time windows (e.g., America/New_York).")
custom_tz_str = ""
target_tz_str = "" # Initialize
if selected_tz_option == "Custom":
    custom_tz_str = st.sidebar.text_input("Enter Custom Timezone (IANA Format)", placeholder="e.g., Europe/Paris")
    target_tz_str = custom_tz_str
else:
    target_tz_str = tz_shortcuts.get(selected_tz_option)


# Day of Week Filter
day_filter_enabled = st.sidebar.toggle("Filter by Day of Week?", value=False)
target_day_name = None
if day_filter_enabled:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] # Include all days
    target_day_name = st.sidebar.selectbox("Select Day", options=days, index=0) # Default to Monday


# Initial Range Type Filter (NEW)
range_type_options = ["All", "Bullish", "Bearish"]
selected_range_type = st.sidebar.selectbox(
    "Filter by Initial Range Type?",
    options=range_type_options,
    index=0, # Default to "All"
    help="Filter days: Bullish, Bearish during initial range."
)


# Date Range Filter
date_filter_enabled = st.sidebar.toggle("Filter by Date Range?", value=True)
start_date = None
end_date = None
if date_filter_enabled:
    # Use reasonable defaults relative to today
    today = date.today()
    default_start_date = today - timedelta(days=365) # Default to 1 year ago
    default_end_date = today
    start_date = st.sidebar.date_input("Analysis Start Date", value=default_start_date)
    end_date = st.sidebar.date_input("Analysis End Date", value=default_end_date)


# Analysis Button
run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

# --- Main Area for Status and Results ---
st.divider() # Visual separator

if run_button:
    # --- Input Validation ---
    st.subheader("Running Analysis...")
    valid_input = True
    user_timezone = None
    error_messages = []

    # Validate Time Inputs
    if not range_start_t or not range_end_t or not post_range_end_t:
        error_messages.append("❌ Please provide all time inputs.")
        valid_input = False
    elif range_end_t <= range_start_t:
        error_messages.append("❌ Initial Range End Time must be after Start Time.")
        valid_input = False
    elif post_range_end_t <= range_end_t:
        error_messages.append("❌ Post-Range End Time must be after Initial Range End Time.")
        valid_input = False

    # Validate Timezone
    if not target_tz_str:
         error_messages.append("❌ Please select or enter a valid timezone.")
         valid_input = False
    else:
         try:
             user_timezone = ZoneInfo(target_tz_str)
         except ZoneInfoNotFoundError:
             error_messages.append(f"❌ Timezone '{target_tz_str}' not found. Use IANA format (e.g., America/New_York).")
             valid_input = False
         except Exception as e:
             error_messages.append(f"❌ Error validating timezone '{target_tz_str}': {e}")
             valid_input = False

    # Validate Date Range (if enabled)
    if date_filter_enabled:
        if not start_date or not end_date:
             error_messages.append("❌ Please provide start and end dates for filtering.")
             valid_input = False
        elif end_date < start_date:
             error_messages.append("❌ End Date must be on or after Start Date.")
             valid_input = False

    # Display validation errors if any and stop
    if not valid_input:
        for msg in error_messages:
            st.error(msg)
    else:
        # --- Inputs are valid, proceed with loading and analysis ---
        master_df = None
        analysis_completed = False
        results = {} # Dictionary to store results

        try:
            # 1. Load Data (uses cache)
            data_url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv'
            with st.spinner('Loading and preparing data... (cached if previously loaded)'):
                master_df = load_data(data_url)

            if master_df is None or master_df.empty:
                # Error should have been shown in load_data or ConnectionError caught
                raise ValueError("Data loading failed or resulted in empty DataFrame. Cannot proceed.")

            # 2. Filter by Date (if enabled)
            filtered_df = master_df.copy() # Start with a copy
            if date_filter_enabled:
                st.write(f"Filtering data from {start_date} to {end_date}...")
                # Ensure comparison is between date objects
                start_date_dt = pd.Timestamp(start_date)
                end_date_dt = pd.Timestamp(end_date)
                filtered_df = filtered_df[
                    (filtered_df.index.normalize() >= start_date_dt) &
                    (filtered_df.index.normalize() <= end_date_dt)
                ]
                if filtered_df.empty:
                    st.warning(f"⚠️ No data found within the specified date range {start_date} to {end_date}. Cannot proceed.")
                    raise ValueError("No data in date range") # Stop processing

            # 3. Timezone Conversion
            st.write(f"Applying timezone conversion to {user_timezone.key}...")
            if filtered_df.index.tz is None:
                # Localize from presumed UTC, handle DST carefully
                st.write("-> Localizing naive timestamps to UTC...")
                filtered_df.index = filtered_df.index.tz_localize('UTC', ambiguous='infer', nonexistent='NaT')
            elif str(filtered_df.index.tz) != 'UTC':
                # If already localized but not UTC, convert to UTC first
                st.write(f"-> Converting existing timezone ({filtered_df.index.tz}) to UTC...")
                filtered_df.index = filtered_df.index.tz_convert('UTC')

            # Convert from UTC to the target user timezone
            st.write(f"-> Converting UTC to target timezone: {user_timezone.key}...")
            filtered_df = filtered_df.tz_convert(user_timezone)
            st.write("Timezone conversion complete.")

            # Handle NaT values potentially created during localization/conversion
            original_len = len(filtered_df)
            # --- CORRECTED NaT Index Handling ---
            # Keep only rows where the index is NOT NaT (Not a Time)
            filtered_df = filtered_df[filtered_df.index.notna()]
            # --- End Correction ---
            dropped_rows = original_len - len(filtered_df)
            if dropped_rows > 0:
                st.warning(f"⚠️ Dropped {dropped_rows} rows with invalid timestamps (NaT) likely created during timezone conversion (e.g., DST).")

            if filtered_df.empty:
                st.warning(f"⚠️ No data remaining after timezone conversion and NaT removal.")
                raise ValueError("No data after TZ conversion")

            # 4. Run Core Analysis
            st.write("Running RTO analysis...")
            # Format times as HH:MM:SS strings for the analysis function
            range_start_str = range_start_t.strftime('%H:%M:%S')
            range_end_str = range_end_t.strftime('%H:%M:%S')
            post_range_end_str = post_range_end_t.strftime('%H:%M:%S')

            with st.spinner("Analyzing days... This may take a moment."):
                (days_processed, breakout_days,
                 b_w_pre, post_g_pre, b_no_pre, post_g_no_pre,
                 intervals_pre, intervals_no_pre) = analyze_dynamic_range_rto(
                     filtered_df, range_start_str, range_end_str, post_range_end_str,
                     target_day_name, user_timezone,
                     selected_range_type # Pass the new filter value
                 )
            analysis_completed = True

            # Store results for display
            results = {
                "days_processed": days_processed, "breakout_days": breakout_days,
                "b_w_pre": b_w_pre, "post_g_pre": post_g_pre,
                "b_no_pre": b_no_pre, "post_g_no_pre": post_g_no_pre,
                "intervals_pre": intervals_pre, "intervals_no_pre": intervals_no_pre,
                "params": { # Store params for easy display
                    "range_start": range_start_str, "range_end": range_end_str, "post_end": post_range_end_str,
                    "timezone": user_timezone.key,
                    "day_filter": target_day_name or "All Applicable",
                    "date_filter": f"{start_date} to {end_date}" if date_filter_enabled else "All Dates Available",
                    "range_type_filter": selected_range_type # Store the filter used
                }
            }
            st.success("✅ Analysis complete!")

        except (ConnectionError, ValueError) as data_err:
             # Catch data loading/filtering/conversion errors specifically
             st.error(f"❌ Data Error: {data_err}")
        except (AmbiguousTimeError, NonExistentTimeError) as tz_err:
             st.error(f"❌ Timezone Error during conversion: {tz_err}. This often occurs around DST transitions. Try adjusting the date range slightly or using UTC timezone.")
        except Exception as e:
             st.error(f"❌ An unexpected error occurred during analysis: {e}")
             st.exception(e) # Show traceback in app for debugging

        # --- Display Results ---
        st.divider() # Separator before results
        if analysis_completed and results: # Check if results exist
            st.header("📊 Analysis Results")
            st.markdown("---")
            st.subheader("Analysis Summary")

            # Display Parameters Used (using 3 columns)
            p = results['params']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Time & Data:**")
                st.write(f"Date Range: {p['date_filter']}")
                st.write(f"Day(s): {p['day_filter']}")
                st.write(f"Timezone: {p['timezone']}")
            with col2:
                st.markdown("**Range & RTO:**")
                # Display times without seconds for cleaner look if they are :00
                range_start_disp = p['range_start'][:-3] if p['range_start'].endswith(':00') else p['range_start']
                range_end_disp = p['range_end'][:-3] if p['range_end'].endswith(':00') else p['range_end']
                post_end_disp = p['post_end'][:-3] if p['post_end'].endswith(':00') else p['post_end']
                st.write(f"Initial Range: {range_start_disp} - <{range_end_disp}")
                st.write(f"RTO Reference: Open @ {range_end_disp} bar")
                st.write(f"Post-Range Window: {range_end_disp} - <{post_end_disp}")
            with col3:
                 st.markdown("**Filters Applied:**")
                 st.write(f"Initial Range Type: {p['range_type_filter']}") # Display the new filter
                 # Add future filters here

            st.markdown("---")
            st.subheader("Overall Counts")
            if results['days_processed'] == 0:
                 st.warning("⚠️ No days matched the selected filters and had data in the initial range. Cannot calculate statistics.")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Days Processed", results['days_processed'], help="Days matching ALL filters w/ data in initial range")
                col2.metric("Days with Breakout", results['breakout_days'])

                st.markdown("---")
                st.subheader("Conditional RTO Probabilities")

                total_post_rto = results['post_g_pre'] + results['post_g_no_pre']
                overall_prob = (total_post_rto / results['breakout_days'] * 100) if results['breakout_days'] > 0 else 0
                st.metric("Overall Post-Breakout RTO Probability", f"{overall_prob:.2f}%", f"Based on {total_post_rto} RTOs / {results['breakout_days']} Breakouts")
                st.markdown("---") # Mini separator

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pre-Breakout RTO Occurred:**")
                    prob_post_given_pre = (results['post_g_pre'] / results['b_w_pre'] * 100) if results['b_w_pre'] > 0 else 0
                    delta_pre = f"{results['post_g_pre']} Post-RTOs / {results['b_w_pre']} Pre-RTO Days"
                    st.metric("P(Post-RTO | Pre-RTO)", f"{prob_post_given_pre:.2f}%", delta_pre, delta_color="off")
                with col2:
                    st.markdown("**NO Pre-Breakout RTO Occurred:**")
                    prob_post_given_no_pre = (results['post_g_no_pre'] / results['b_no_pre'] * 100) if results['b_no_pre'] > 0 else 0
                    delta_no_pre = f"{results['post_g_no_pre']} Post-RTOs / {results['b_no_pre']} No Pre-RTO Days"
                    st.metric("P(Post-RTO | No Pre-RTO)", f"{prob_post_given_no_pre:.2f}%", delta_no_pre, delta_color="off")


                st.markdown("---")
                st.subheader("Interval Distribution of Post-Breakout RTO")

                # Calculate interval labels (using the actual time objects selected by user)
                # Use the originally selected time objects, not the string versions
                interval_labels = get_interval_labels(range_end_t, post_range_end_t)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pre-Breakout RTO Occurred:**")
                    total_post_given_pre = results['post_g_pre'] # Use the count of post-RTOs given pre-RTO as the denominator
                    if total_post_given_pre > 0:
                        interval_data_pre = []
                        for i in range(5):
                            count = results['intervals_pre'][i]
                            # Calculate probability based on the number of times a post-RTO happened in this condition
                            prob = (count / total_post_given_pre) * 100
                            interval_data_pre.append({"Interval": interval_labels[i], "Count": count, "Probability (%)": f"{prob:.2f}"})
                        st.dataframe(pd.DataFrame(interval_data_pre).set_index("Interval"), use_container_width=True)
                        st.caption(f"Total Post-RTOs (given Pre-RTO occurred): {total_post_given_pre}")
                    else:
                        st.write("(No days with Post-Breakout RTO given Pre-Breakout RTO occurred)")

                with col2:
                    st.markdown("**NO Pre-Breakout RTO Occurred:**")
                    total_post_given_no_pre = results['post_g_no_pre'] # Use the count of post-RTOs given no pre-RTO as the denominator
                    if total_post_given_no_pre > 0:
                        interval_data_no_pre = []
                        for i in range(5):
                            count = results['intervals_no_pre'][i]
                             # Calculate probability based on the number of times a post-RTO happened in this condition
                            prob = (count / total_post_given_no_pre) * 100
                            interval_data_no_pre.append({"Interval": interval_labels[i], "Count": count, "Probability (%)": f"{prob:.2f}"})
                        st.dataframe(pd.DataFrame(interval_data_no_pre).set_index("Interval"), use_container_width=True)
                        st.caption(f"Total Post-RTOs (given No Pre-RTO occurred): {total_post_given_no_pre}")
                    else:
                        st.write("(No days with Post-Breakout RTO given NO Pre-Breakout RTO occurred)")

        elif not valid_input: # Case where run failed due to input validation
             pass # Errors already shown above
        else: # Case where analysis failed mid-way or didn't produce results
             st.warning("Analysis did not complete successfully or produced no results. Please check parameters and data validity.")

else:
    st.info("ℹ️ Configure parameters in the sidebar and click 'Run Analysis'.")