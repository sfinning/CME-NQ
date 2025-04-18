# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import time, datetime, timedelta, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import sys
import math
from pytz import AmbiguousTimeError, NonExistentTimeError

# =============================================================================
# Core Analysis Function
# =============================================================================
# MODIFIED: Added initial_range_type_filter parameter
# MODIFIED: Added breakout direction categorization (Bullish/Bearish)
def analyze_dynamic_range_rto(df, range_start_str, range_end_str, post_range_end_str, target_day_name, user_timezone, initial_range_type_filter="All"):
    """
    Analyzes price action, tracking RTO events.
    MODIFIED: Tracks Post-Breakout RTO interval occurrences separately based on
              whether a Pre-Breakout RTO occurred on the same day AND
              whether the breakout was Bullish or Bearish.
    MODIFIED: Post-Breakout RTO search now starts strictly AFTER the breakout bar.
    FIXED: Check for empty unique_days_in_data using .empty attribute.
    NEW: Filters days based on whether the initial range was Bullish or Bearish.
    NEW: Categorizes results by breakout direction (Bullish/Bearish).

    Returns: A dictionary containing all calculated counts and interval lists.
             Keys include: 'matching_days_processed', 'breakout_days_count',
             'bull_breakout_days', 'bear_breakout_days',
             'bull_b_w_pre', 'bull_post_g_pre', 'bull_b_no_pre', 'bull_post_g_no_pre',
             'bear_b_w_pre', 'bear_post_g_pre', 'bear_b_no_pre', 'bear_post_g_no_pre',
             'bull_intervals_pre', 'bull_intervals_no_pre',
             'bear_intervals_pre', 'bear_intervals_no_pre'
    """
    # --- Initial setup ---
    # --- Initialize Counters ---
    results = {
        "matching_days_processed": 0,
        "breakout_days_count": 0,
        # Bullish Breakout Counts
        "bull_breakout_days": 0,
        "bull_b_w_pre": 0,          # Breakout With Pre-RTO (Bullish)
        "bull_post_g_pre": 0,       # Post-RTO Given Pre-RTO (Bullish)
        "bull_b_no_pre": 0,         # Breakout No Pre-RTO (Bullish)
        "bull_post_g_no_pre": 0,    # Post-RTO Given No Pre-RTO (Bullish)
        "bull_intervals_pre": [0] * 5, # Post-RTO Intervals Given Pre-RTO (Bullish)
        "bull_intervals_no_pre": [0] * 5,# Post-RTO Intervals Given No Pre-RTO (Bullish)
        # Bearish Breakout Counts
        "bear_breakout_days": 0,
        "bear_b_w_pre": 0,          # Breakout With Pre-RTO (Bearish)
        "bear_post_g_pre": 0,       # Post-RTO Given Pre-RTO (Bearish)
        "bear_b_no_pre": 0,         # Breakout No Pre-RTO (Bearish)
        "bear_post_g_no_pre": 0,    # Post-RTO Given No Pre-RTO (Bearish)
        "bear_intervals_pre": [0] * 5, # Post-RTO Intervals Given Pre-RTO (Bearish)
        "bear_intervals_no_pre": [0] * 5 # Post-RTO Intervals Given No Pre-RTO (Bearish)
    }
    # ---

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
             st.warning("Post-range duration is zero or negative. Check time inputs.")
             return results # Return initial zeroed results
        interval_duration = total_post_range_duration / 5
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid time format/inputs for interval calculation: {e}")

    # --- Filter data by day name if specified ---
    if target_day_name:
        data_to_process = df[df.index.day_name() == target_day_name]
        if data_to_process.empty:
            st.warning(f"No data found for the selected day: {target_day_name}")
            return results
    else:
        data_to_process = df
        if data_to_process.empty:
             raise ValueError("Input DataFrame to analyze_dynamic_range_rto is empty.")

    unique_days_in_data = data_to_process.index.normalize().unique()

    if unique_days_in_data.empty:
        st.warning("No unique days found in the filtered data to process.")
        return results

    # --- Process each unique day ---
    for day_date in unique_days_in_data:
        day_data = data_to_process[data_to_process.index.date == day_date.date()]
        if day_data.empty: continue

        current_day_date_part = day_date.date()
        interval_boundaries_day = []
        try:
            range_start_dt = datetime.combine(current_day_date_part, range_start_t, tzinfo=user_timezone)
            range_end_dt = datetime.combine(current_day_date_part, range_end_t, tzinfo=user_timezone)
            post_range_end_dt_day = datetime.combine(current_day_date_part, post_range_end_t, tzinfo=user_timezone)
            range_end_exclusive_dt = range_end_dt - timedelta(microseconds=1)
            post_range_end_exclusive_dt_day = post_range_end_dt_day - timedelta(microseconds=1)

            current_boundary_dt = range_end_dt
            for i in range(5):
                end_dt_calc = current_boundary_dt + interval_duration
                next_boundary_dt = min(end_dt_calc, post_range_end_dt_day)
                if i == 4: next_boundary_dt = post_range_end_dt_day
                interval_boundaries_day.append((current_boundary_dt, next_boundary_dt))
                current_boundary_dt = next_boundary_dt
        except Exception as e:
            continue # Skip day on timezone/combine issues

        # --- 1. Analyze Initial Range ---
        initial_range_data = day_data.loc[range_start_dt:range_end_exclusive_dt]
        if initial_range_data.empty: continue

        # ===> Initial Range Type Filter Logic <===
        if initial_range_type_filter != "All":
            try:
                initial_open = initial_range_data['Open'].iloc[0]
                initial_close = initial_range_data['Close'].iloc[-1]
                is_bullish_range = initial_close > initial_open
                is_bearish_range = initial_close < initial_open
                if initial_range_type_filter == "Bullish" and not is_bullish_range: continue
                if initial_range_type_filter == "Bearish" and not is_bearish_range: continue
            except IndexError:
                continue
        # ===> END FILTER LOGIC <===

        results["matching_days_processed"] += 1
        range_high = initial_range_data['High'].max()
        range_low = initial_range_data['Low'].min()

        # --- 2. Get Reference Price & Post-Range Data ---
        post_range_data = day_data.loc[range_end_dt:post_range_end_exclusive_dt_day]
        if post_range_data.empty: continue
        try:
            reference_open_price = post_range_data['Open'].iloc[0]
        except IndexError:
            continue

        # --- 3. Find First Breakout & Direction ---
        first_breakout_time = None
        is_bullish_breakout = None # None: No breakout, True: Bullish, False: Bearish

        high_break_condition = post_range_data['High'] > range_high
        first_high_break_time = post_range_data[high_break_condition].index.min() if high_break_condition.any() else None
        low_break_condition = post_range_data['Low'] < range_low
        first_low_break_time = post_range_data[low_break_condition].index.min() if low_break_condition.any() else None

        if first_high_break_time and first_low_break_time:
            if first_high_break_time <= first_low_break_time: # Prioritize high break on exact tie
                first_breakout_time = first_high_break_time
                is_bullish_breakout = True
            else:
                first_breakout_time = first_low_break_time
                is_bullish_breakout = False
        elif first_high_break_time:
            first_breakout_time = first_high_break_time
            is_bullish_breakout = True
        elif first_low_break_time:
            first_breakout_time = first_low_break_time
            is_bullish_breakout = False

        # --- 4. Check for Pre-RTO, Post-RTO & Determine Interval ---
        if first_breakout_time is not None: # Equivalent to checking if is_bullish_breakout is not None
            results["breakout_days_count"] += 1
            if is_bullish_breakout:
                results["bull_breakout_days"] += 1
            else:
                results["bear_breakout_days"] += 1

            pre_rto_occurred = False
            post_rto_occurred = False
            first_post_rto_time = None

            # --- 4a. Check for Pre-Breakout RTO ---
            pre_breakout_check_data = post_range_data[post_range_data.index < first_breakout_time]
            if not pre_breakout_check_data.empty:
                pre_rto_condition = (pre_breakout_check_data['Low'] <= reference_open_price) & (pre_breakout_check_data['High'] >= reference_open_price)
                if pre_rto_condition.any():
                    pre_rto_occurred = True
                    # Increment pre-RTO counts based on breakout direction
                    if is_bullish_breakout:
                        results["bull_b_w_pre"] += 1
                    else:
                        results["bear_b_w_pre"] += 1
                else:
                    # Increment no pre-RTO counts based on breakout direction
                    if is_bullish_breakout:
                        results["bull_b_no_pre"] += 1
                    else:
                        results["bear_b_no_pre"] += 1
            else:
                # No bars between range end and breakout -> No Pre-RTO
                if is_bullish_breakout:
                    results["bull_b_no_pre"] += 1
                else:
                    results["bear_b_no_pre"] += 1

            # --- 4b. Check for Post-Breakout RTO ---
            rto_check_data = post_range_data[post_range_data.index > first_breakout_time]
            if not rto_check_data.empty:
                post_rto_condition = (rto_check_data['Low'] <= reference_open_price) & (rto_check_data['High'] >= reference_open_price)
                if post_rto_condition.any():
                     try:
                         first_post_rto_time = rto_check_data[post_rto_condition].index.min()
                         post_rto_occurred = True
                     except ValueError: pass

            # --- 4c. Increment Conditional Counters & Interval Counts ---
            if post_rto_occurred:
                if pre_rto_occurred:
                    # Increment Post-RTO given Pre-RTO counts based on breakout direction
                    if is_bullish_breakout:
                        results["bull_post_g_pre"] += 1
                    else:
                        results["bear_post_g_pre"] += 1
                else:
                    # Increment Post-RTO given No Pre-RTO counts based on breakout direction
                    if is_bullish_breakout:
                        results["bull_post_g_no_pre"] += 1
                    else:
                        results["bear_post_g_no_pre"] += 1

                # Determine interval and increment based on pre-RTO status AND breakout direction
                if first_post_rto_time:
                    for i in range(5):
                        start_interval, end_interval = interval_boundaries_day[i]
                        is_in_interval = False
                        if i == 4: # Last interval includes end boundary
                             if start_interval <= first_post_rto_time <= end_interval: is_in_interval = True
                        else: # Other intervals are [start, end)
                             if start_interval <= first_post_rto_time < end_interval: is_in_interval = True

                        if is_in_interval:
                            if pre_rto_occurred:
                                if is_bullish_breakout:
                                    results["bull_intervals_pre"][i] += 1
                                else:
                                    results["bear_intervals_pre"][i] += 1
                            else: # No pre-RTO
                                if is_bullish_breakout:
                                    results["bull_intervals_no_pre"][i] += 1
                                else:
                                    results["bear_intervals_no_pre"][i] += 1
                            break # RTO only in one interval
        # --- End of processing for a single day ---

    # Return the dictionary containing all results
    return results

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

            # Format times as HH:MM
            start_t_str = current_interval_start_dt.strftime('%H:%M')
            end_t_str = current_interval_end_dt.strftime('%H:%M')

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
st.markdown("Analyze the probability and timing of price returning to the open price after a breakout.")

# --- Sidebar for Inputs ---
st.sidebar.header("Analysis Parameters")

# Time Windows
default_start_time = time(9, 0)
default_end_time = time(10, 0)
default_post_end_time = time(11, 0)
range_start_t = st.sidebar.time_input("Initial Range Start Time", value=default_start_time, help="Start of the initial balance range (inclusive).", step=60)
range_end_t = st.sidebar.time_input("Initial Range End Time", value=default_end_time, help="End of the initial balance range. Reference price is Open of this bar.", step=60)
post_range_end_t = st.sidebar.time_input("Post-Range End Time", value=default_post_end_time, help="End of the window to check for breakouts and RTOs.", step=60)

# Timezone Input
tz_shortcuts = {"EST": "America/New_York", "CST": "America/Chicago", "UTC": "UTC"}
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
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    target_day_name = st.sidebar.selectbox("Select Day", options=days, index=0)


# Initial Range Type Filter
range_type_options = ["All", "Bullish", "Bearish"]
selected_range_type = st.sidebar.selectbox(
    "Filter by Initial Range Bias?",
    options=range_type_options,
    index=0,
    help="Filter days: Bullish (Close > Open), Bearish (Close < Open) during initial range."
)


# Date Range Filter
date_filter_enabled = st.sidebar.toggle("Filter by Date Range?", value=True)
start_date = None
end_date = None
if date_filter_enabled:
    today = date.today()
    default_start_date = today - timedelta(days=365)
    default_end_date = today
    start_date = st.sidebar.date_input("Analysis Start Date", value=default_start_date)
    end_date = st.sidebar.date_input("Analysis End Date", value=default_end_date)


# Analysis Button
run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

# --- Main Area for Status and Results ---
st.divider()

if run_button:
    # --- Input Validation ---
    st.subheader("Running Analysis...")
    valid_input = True
    user_timezone = None
    error_messages = []

    if not range_start_t or not range_end_t or not post_range_end_t:
        error_messages.append("❌ Please provide all time inputs.")
        valid_input = False
    elif range_end_t <= range_start_t:
        error_messages.append("❌ Initial Range End Time must be after Start Time.")
        valid_input = False
    elif post_range_end_t <= range_end_t:
        error_messages.append("❌ Post-Range End Time must be after Initial Range End Time.")
        valid_input = False

    if not target_tz_str:
         error_messages.append("❌ Please select or enter a valid timezone.")
         valid_input = False
    else:
         try: user_timezone = ZoneInfo(target_tz_str)
         except ZoneInfoNotFoundError:
             error_messages.append(f"❌ Timezone '{target_tz_str}' not found. Use IANA format.")
             valid_input = False
         except Exception as e:
             error_messages.append(f"❌ Error validating timezone '{target_tz_str}': {e}")
             valid_input = False

    if date_filter_enabled:
        if not start_date or not end_date:
             error_messages.append("❌ Please provide start and end dates for filtering.")
             valid_input = False
        elif end_date < start_date:
             error_messages.append("❌ End Date must be on or after Start Date.")
             valid_input = False

    if not valid_input:
        for msg in error_messages: st.error(msg)
    else:
        # --- Inputs valid, proceed ---
        master_df = None
        analysis_completed = False
        results = {} # Dictionary to store results

        try:
            # 1. Load Data
            data_url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv'
            with st.spinner('Loading and preparing data... (cached)'):
                master_df = load_data(data_url)

            if master_df is None or master_df.empty:
                raise ValueError("Data loading failed or resulted in empty DataFrame.")

            # 2. Filter by Date
            filtered_df = master_df.copy()
            if date_filter_enabled:
                st.write(f"Filtering data from {start_date} to {end_date}...")
                start_date_dt = pd.Timestamp(start_date)
                end_date_dt = pd.Timestamp(end_date)
                filtered_df = filtered_df[
                    (filtered_df.index.normalize() >= start_date_dt) &
                    (filtered_df.index.normalize() <= end_date_dt)
                ]
                if filtered_df.empty:
                    st.warning(f"⚠️ No data found within the date range {start_date} to {end_date}.")
                    raise ValueError("No data in date range")

            # 3. Timezone Conversion
            st.write(f"Applying timezone conversion to {user_timezone.key}...")
            if filtered_df.index.tz is None:
                st.write("-> Localizing naive timestamps to UTC...")
                filtered_df.index = filtered_df.index.tz_localize('UTC', ambiguous='infer', nonexistent='NaT')
            elif str(filtered_df.index.tz) != 'UTC':
                st.write(f"-> Converting existing timezone ({filtered_df.index.tz}) to UTC...")
                filtered_df.index = filtered_df.index.tz_convert('UTC')

            st.write(f"-> Converting UTC to target timezone: {user_timezone.key}...")
            filtered_df = filtered_df.tz_convert(user_timezone)

            original_len = len(filtered_df)
            filtered_df = filtered_df[filtered_df.index.notna()]
            dropped_rows = original_len - len(filtered_df)
            if dropped_rows > 0:
                st.warning(f"⚠️ Dropped {dropped_rows} rows with invalid timestamps (NaT) during timezone conversion (e.g., DST).")

            if filtered_df.empty:
                st.warning(f"⚠️ No data remaining after timezone conversion and NaT removal.")
                raise ValueError("No data after TZ conversion")

            # 4. Run Core Analysis
            st.write("Running RTO analysis...")
            range_start_str = range_start_t.strftime('%H:%M:%S')
            range_end_str = range_end_t.strftime('%H:%M:%S')
            post_range_end_str = post_range_end_t.strftime('%H:%M:%S')

            with st.spinner("Analyzing days... This may take a moment."):
                # Call the updated analysis function which returns a dictionary
                results = analyze_dynamic_range_rto(
                     filtered_df, range_start_str, range_end_str, post_range_end_str,
                     target_day_name, user_timezone,
                     selected_range_type
                 )
            analysis_completed = True

            # Store parameters used in the results dictionary for display
            results["params"] = {
                "range_start": range_start_str, "range_end": range_end_str, "post_end": post_range_end_str,
                "timezone": user_timezone.key,
                "day_filter": target_day_name or "All Applicable",
                "date_filter": f"{start_date} to {end_date}" if date_filter_enabled else "All Dates Available",
                "range_type_filter": selected_range_type
            }
            st.success("✅ Analysis complete!")

        except (ConnectionError, ValueError) as data_err:
             st.error(f"❌ Data Error: {data_err}")
        except (AmbiguousTimeError, NonExistentTimeError) as tz_err:
             st.error(f"❌ Timezone Error during conversion: {tz_err}. Often occurs around DST. Try adjusting date range or using UTC.")
        except Exception as e:
             st.error(f"❌ An unexpected error occurred during analysis: {e}")
             st.exception(e)

        # --- Display Results ---
        st.divider()
        if analysis_completed and results and results.get("matching_days_processed") is not None: # Check results exist and have expected keys
            st.header("📊 Analysis Results")
            st.markdown("---")
            st.subheader("Analysis Summary")

            # Display Parameters Used
            p = results['params']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Time & Data:**")
                st.write(f"Date Range: {p['date_filter']}")
                st.write(f"Day(s): {p['day_filter']}")
                st.write(f"Timezone: {p['timezone']}")
            with col2:
                st.markdown("**Range & RTO:**")
                range_start_disp = p['range_start'][:-3] if p['range_start'].endswith(':00') else p['range_start']
                range_end_disp = p['range_end'][:-3] if p['range_end'].endswith(':00') else p['range_end']
                post_end_disp = p['post_end'][:-3] if p['post_end'].endswith(':00') else p['post_end']
                st.write(f"Initial Range: {range_start_disp} - <{range_end_disp}")
                st.write(f"RTO Reference: Open @ {range_end_disp} bar")
                st.write(f"Post-Range Window: {range_end_disp} - <{post_end_disp}")
            with col3:
                 st.markdown("**Filters Applied:**")
                 st.write(f"Initial Range Type: {p['range_type_filter']}")

            st.markdown("---")
            st.subheader("Overall Counts")
            if results['matching_days_processed'] == 0:
                 st.warning("⚠️ No days matched the selected filters and had data in the initial range. Cannot calculate statistics.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Days Processed", results['matching_days_processed'], help="Days matching ALL filters w/ data in initial range")
                col2.metric("Total Breakout Days", results['breakout_days_count'])
                # Add breakout direction counts
                col3.metric("Bullish Breakouts", results['bull_breakout_days'])
                col3.metric("Bearish Breakouts", results['bear_breakout_days'])


                st.markdown("---")
                st.subheader("Conditional RTO Probabilities by Breakout Direction")

                # Use tabs for Bullish vs Bearish Breakouts
                tab1, tab2 = st.tabs(["🐂 Bullish Breakouts", "🐻 Bearish Breakouts"])

                with tab1: # Bullish Breakouts
                    st.markdown(f"**Total Bullish Breakout Days: {results['bull_breakout_days']}**")
                    if results['bull_breakout_days'] > 0:
                        col1_bull, col2_bull = st.columns(2)
                        with col1_bull:
                            st.markdown("**Pre-Breakout RTO Occurred:**")
                            prob_post_given_pre_bull = (results['bull_post_g_pre'] / results['bull_b_w_pre'] * 100) if results['bull_b_w_pre'] > 0 else 0
                            delta_pre_bull = f"{results['bull_post_g_pre']} Post-RTOs / {results['bull_b_w_pre']} Pre-RTO Days"
                            st.metric("P(Post-RTO | Pre-RTO, Bullish BO)", f"{prob_post_given_pre_bull:.2f}%", delta_pre_bull, delta_color="off")
                        with col2_bull:
                            st.markdown("**NO Pre-Breakout RTO Occurred:**")
                            prob_post_given_no_pre_bull = (results['bull_post_g_no_pre'] / results['bull_b_no_pre'] * 100) if results['bull_b_no_pre'] > 0 else 0
                            delta_no_pre_bull = f"{results['bull_post_g_no_pre']} Post-RTOs / {results['bull_b_no_pre']} No Pre-RTO Days"
                            st.metric("P(Post-RTO | No Pre-RTO, Bullish BO)", f"{prob_post_given_no_pre_bull:.2f}%", delta_no_pre_bull, delta_color="off")
                    else:
                        st.write("(No bullish breakout days found matching criteria)")

                with tab2: # Bearish Breakouts
                    st.markdown(f"**Total Bearish Breakout Days: {results['bear_breakout_days']}**")
                    if results['bear_breakout_days'] > 0:
                        col1_bear, col2_bear = st.columns(2)
                        with col1_bear:
                            st.markdown("**Pre-Breakout RTO Occurred:**")
                            prob_post_given_pre_bear = (results['bear_post_g_pre'] / results['bear_b_w_pre'] * 100) if results['bear_b_w_pre'] > 0 else 0
                            delta_pre_bear = f"{results['bear_post_g_pre']} Post-RTOs / {results['bear_b_w_pre']} Pre-RTO Days"
                            st.metric("P(Post-RTO | Pre-RTO, Bearish BO)", f"{prob_post_given_pre_bear:.2f}%", delta_pre_bear, delta_color="off")
                        with col2_bear:
                            st.markdown("**NO Pre-Breakout RTO Occurred:**")
                            prob_post_given_no_pre_bear = (results['bear_post_g_no_pre'] / results['bear_b_no_pre'] * 100) if results['bear_b_no_pre'] > 0 else 0
                            delta_no_pre_bear = f"{results['bear_post_g_no_pre']} Post-RTOs / {results['bear_b_no_pre']} No Pre-RTO Days"
                            st.metric("P(Post-RTO | No Pre-RTO, Bearish BO)", f"{prob_post_given_no_pre_bear:.2f}%", delta_no_pre_bear, delta_color="off")
                    else:
                        st.write("(No bearish breakout days found matching criteria)")


                st.markdown("---")
                st.subheader("Distribution of RTO Timing")

                interval_labels = get_interval_labels(range_end_t, post_range_end_t)

                # Display interval distributions, split by Pre-RTO status and then by Breakout Direction
                col_dist_1, col_dist_2 = st.columns(2)

                with col_dist_1:
                    st.markdown("**Condition: Pre-Breakout RTO Occurred**")
                    st.markdown("**Sub-Condition: Bullish Breakout**")
                    total_post_given_pre_bull = results['bull_post_g_pre']
                    if total_post_given_pre_bull > 0:
                        interval_data = []
                        for i in range(5):
                            count = results['bull_intervals_pre'][i]
                            prob = (count / total_post_given_pre_bull) * 100
                            interval_data.append({"Interval": interval_labels[i], "Count": count, "Probability (%)": f"{prob:.2f}"})
                        st.dataframe(pd.DataFrame(interval_data).set_index("Interval"), use_container_width=True)
                        st.caption(f"Total Post-RTOs (Pre-RTO, Bullish BO): {total_post_given_pre_bull}")
                    else:
                        st.write("(No Post-RTOs under this condition)")

                    st.markdown("**Sub-Condition: Bearish Breakout**")
                    total_post_given_pre_bear = results['bear_post_g_pre']
                    if total_post_given_pre_bear > 0:
                        interval_data = []
                        for i in range(5):
                            count = results['bear_intervals_pre'][i]
                            prob = (count / total_post_given_pre_bear) * 100
                            interval_data.append({"Interval": interval_labels[i], "Count": count, "Probability (%)": f"{prob:.2f}"})
                        st.dataframe(pd.DataFrame(interval_data).set_index("Interval"), use_container_width=True)
                        st.caption(f"Total Post-RTOs (Pre-RTO, Bearish BO): {total_post_given_pre_bear}")
                    else:
                        st.write("(No Post-RTOs under this condition)")


                with col_dist_2:
                    st.markdown("**Condition: NO Pre-Breakout RTO Occurred**")
                    st.markdown("**Sub-Condition: Bullish Breakout**")
                    total_post_given_no_pre_bull = results['bull_post_g_no_pre']
                    if total_post_given_no_pre_bull > 0:
                        interval_data = []
                        for i in range(5):
                            count = results['bull_intervals_no_pre'][i]
                            prob = (count / total_post_given_no_pre_bull) * 100
                            interval_data.append({"Interval": interval_labels[i], "Count": count, "Probability (%)": f"{prob:.2f}"})
                        st.dataframe(pd.DataFrame(interval_data).set_index("Interval"), use_container_width=True)
                        st.caption(f"Total Post-RTOs (No Pre-RTO, Bullish BO): {total_post_given_no_pre_bull}")
                    else:
                        st.write("(No Post-RTOs under this condition)")

                    st.markdown("**Sub-Condition: Bearish Breakout**")
                    total_post_given_no_pre_bear = results['bear_post_g_no_pre']
                    if total_post_given_no_pre_bear > 0:
                        interval_data = []
                        for i in range(5):
                            count = results['bear_intervals_no_pre'][i]
                            prob = (count / total_post_given_no_pre_bear) * 100
                            interval_data.append({"Interval": interval_labels[i], "Count": count, "Probability (%)": f"{prob:.2f}"})
                        st.dataframe(pd.DataFrame(interval_data).set_index("Interval"), use_container_width=True)
                        st.caption(f"Total Post-RTOs (No Pre-RTO, Bearish BO): {total_post_given_no_pre_bear}")
                    else:
                        st.write("(No Post-RTOs under this condition)")


        elif not valid_input:
             pass # Errors already shown
        else:
             st.warning("Analysis did not complete successfully or produced no results. Check parameters and data validity.")

else:
    st.info("ℹ️ Configure parameters in the sidebar and click 'Run Analysis'.")

