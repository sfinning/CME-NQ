# probability_analyzer.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import time, datetime, timedelta, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import math

# =============================================================================
# Data Loading Function (Adapted from streamlit_app.py)
# =============================================================================
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(url):
    """Loads data from URL, prepares OHLCV columns, and returns DataFrame (UTC)."""
    try:
        st.write(f"Loading data from {url}...") # Show status in app
        temp_df = pd.read_csv(url)

        # Basic Column Validation
        if 'ts_event' not in temp_df.columns or 'symbol' not in temp_df.columns:
            raise ValueError("Essential columns 'ts_event' or 'symbol' missing.")

        # Timestamp Conversion and Indexing (assuming nanoseconds UTC)
        temp_df['ts_event'] = pd.to_datetime(temp_df['ts_event'], unit='ns', errors='coerce')
        temp_df.dropna(subset=['ts_event'], inplace=True) # Drop rows where conversion failed
        # Ensure data is UTC initially
        temp_df = temp_df.set_index('ts_event').sort_index().tz_localize('UTC')

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
# Core Probability Calculation Function
# =============================================================================
def calculate_rto_probability(df, target_days, initial_range_start_hour, prev_hour_state_filter, user_timezone):
    """
    Calculates the probability of RTO based on filtered historical data.

    Args:
        df (pd.DataFrame): The input DataFrame with timezone-aware index.
        target_days (list): List of day names (e.g., ["Monday", "Friday"]).
        initial_range_start_hour (int): The hour (0-23) when the initial range starts.
        prev_hour_state_filter (str): "All", "Bullish", or "Bearish".
        user_timezone (ZoneInfo): The timezone object for calculations.

    Returns:
        tuple: (probability, total_breakout_scenarios, rto_occurred_scenarios)
               Returns (None, 0, 0) if no valid scenarios found or on error.
    """
    total_breakout_scenarios = 0
    rto_occurred_scenarios = 0

    if df is None or df.empty:
        st.warning("Input DataFrame is empty. Cannot calculate probability.")
        return None, 0, 0

    # 1. Filter by Day of Week
    if target_days: # Only filter if list is not empty
        df_filtered_day = df[df.index.day_name().isin(target_days)]
    else: # If no days selected, use all data (or handle as error depending on desired behavior)
        st.warning("No target days selected. Analyzing all available days.")
        df_filtered_day = df # Or return None, 0, 0 if a day must be selected

    if df_filtered_day.empty:
        st.warning(f"No data found for selected day(s): {', '.join(target_days)}")
        return None, 0, 0

    unique_days_in_data = df_filtered_day.index.normalize().unique()
    if unique_days_in_data.empty:
        st.warning("No unique days found in the filtered data.")
        return None, 0, 0

    # --- Define Time Windows based on input hour ---
    try:
        initial_range_start_t = time(initial_range_start_hour, 0)
        # Assuming 1-hour initial range, 1-hour post-range window
        initial_range_end_t = time((initial_range_start_hour + 1) % 24, 0)
        post_range_end_t = time((initial_range_start_hour + 2) % 24, 0)
        prev_hour_start_t = time((initial_range_start_hour - 1 + 24) % 24, 0) # Handle wrap-around
        prev_hour_end_t = initial_range_start_t # End of prev hour is start of initial range

        # Handle overnight ranges (e.g., initial range 23:00-00:00)
        initial_range_crosses_midnight = initial_range_end_t < initial_range_start_t
        post_range_crosses_midnight = post_range_end_t < initial_range_end_t
        prev_hour_crosses_midnight = prev_hour_end_t < prev_hour_start_t

    except Exception as e:
        st.error(f"Error defining time windows: {e}")
        return None, 0, 0

    # --- Process each unique day ---
    for day_date in unique_days_in_data:
        current_day_date_part = day_date.date()
        next_day_date_part = current_day_date_part + timedelta(days=1)
        prev_day_date_part = current_day_date_part - timedelta(days=1)

        try:
            # --- Define Datetime Boundaries for the specific day ---
            # Previous Hour
            prev_hour_start_dt = datetime.combine(prev_day_date_part if prev_hour_crosses_midnight else current_day_date_part, prev_hour_start_t, tzinfo=user_timezone)
            prev_hour_end_dt = datetime.combine(current_day_date_part, prev_hour_end_t, tzinfo=user_timezone)
            prev_hour_end_exclusive_dt = prev_hour_end_dt - timedelta(microseconds=1)

            # Initial Range
            initial_range_start_dt = datetime.combine(current_day_date_part, initial_range_start_t, tzinfo=user_timezone)
            initial_range_end_dt = datetime.combine(next_day_date_part if initial_range_crosses_midnight else current_day_date_part, initial_range_end_t, tzinfo=user_timezone)
            initial_range_end_exclusive_dt = initial_range_end_dt - timedelta(microseconds=1)

            # Post-Range Window (where RTO is checked)
            post_range_start_dt = initial_range_end_dt # Starts exactly when initial range ends
            post_range_end_dt_day = datetime.combine(next_day_date_part if post_range_crosses_midnight else current_day_date_part, post_range_end_t, tzinfo=user_timezone)
            post_range_end_exclusive_dt_day = post_range_end_dt_day - timedelta(microseconds=1)

            # --- 1. Analyze Previous Hour State ---
            prev_hour_data = df_filtered_day.loc[prev_hour_start_dt:prev_hour_end_exclusive_dt]
            if prev_hour_data.empty: continue # Need previous hour data

            try:
                prev_open = prev_hour_data['Open'].iloc[0]
                prev_close = prev_hour_data['Close'].iloc[-1]
                is_bullish_prev = prev_close > prev_open
                is_bearish_prev = prev_close < prev_open
            except IndexError:
                continue # Skip if prev hour data is incomplete

            # Apply Previous Hour State Filter
            if prev_hour_state_filter == "Bullish" and not is_bullish_prev: continue
            if prev_hour_state_filter == "Bearish" and not is_bearish_prev: continue
            # "All" passes through

            # --- 2. Analyze Initial Range ---
            initial_range_data = df_filtered_day.loc[initial_range_start_dt:initial_range_end_exclusive_dt]
            if initial_range_data.empty: continue # Need initial range data

            range_high = initial_range_data['High'].max()
            range_low = initial_range_data['Low'].min()

            # --- 3. Get Reference Price & Post-Range Data ---
            post_range_data = df_filtered_day.loc[post_range_start_dt:post_range_end_exclusive_dt_day]
            if post_range_data.empty: continue # Need post-range data to check for RTO

            try:
                # Reference price is the Open of the *first* bar *after* the initial range ends
                reference_open_price = post_range_data['Open'].iloc[0]
            except IndexError:
                continue # Skip if post-range data is incomplete

            # --- 4. Find First Breakout in Post-Range Window ---
            first_breakout_time = None
            high_break_condition = post_range_data['High'] > range_high
            first_high_break_time = post_range_data[high_break_condition].index.min() if high_break_condition.any() else None
            low_break_condition = post_range_data['Low'] < range_low
            first_low_break_time = post_range_data[low_break_condition].index.min() if low_break_condition.any() else None

            if first_high_break_time and first_low_break_time:
                first_breakout_time = min(first_high_break_time, first_low_break_time)
            elif first_high_break_time:
                first_breakout_time = first_high_break_time
            elif first_low_break_time:
                first_breakout_time = first_low_break_time
            else:
                continue # No breakout occurred in the post-range window for this day/scenario

            # --- 5. Check for Post-Breakout RTO ---
            # Increment total scenarios *only if* a breakout happened under the filtered conditions
            total_breakout_scenarios += 1

            # Check for RTO strictly *after* the breakout bar
            rto_check_data = post_range_data[post_range_data.index > first_breakout_time]
            if not rto_check_data.empty:
                post_rto_condition = (rto_check_data['Low'] <= reference_open_price) & (rto_check_data['High'] >= reference_open_price)
                if post_rto_condition.any():
                    rto_occurred_scenarios += 1 # Increment RTO count for this scenario

        except Exception as e:
            # Log or warn about errors processing a specific day, but continue
            # st.warning(f"Skipping day {day_date.date()} due to error: {e}")
            continue # Skip day on any processing error

    # --- Calculate Final Probability ---
    if total_breakout_scenarios == 0:
        st.warning("No breakout scenarios found matching all filter criteria.")
        return None, 0, 0
    else:
        probability = (rto_occurred_scenarios / total_breakout_scenarios) * 100
        return probability, total_breakout_scenarios, rto_occurred_scenarios


# =============================================================================
# Streamlit Application UI
# =============================================================================

st.set_page_config(layout="centered") # Centered layout might be better for single viz
st.title("📊 RTO Probability Analyzer")
st.markdown("Calculate the historical probability of Return-To-Open (RTO) based on selected filters.")

# --- Sidebar for Inputs ---
st.sidebar.header("Analysis Filters")

# Timezone Input (Crucial for defining hours correctly)
tz_shortcuts = {"EST": "America/New_York", "CST": "America/Chicago", "UTC": "UTC"}
tz_options = list(tz_shortcuts.keys()) + ["Custom"]
selected_tz_option = st.sidebar.selectbox(
    "Select Timezone",
    options=tz_options,
    index=tz_options.index("CST"), # Default to Chicago
    help="Timezone for defining the Hour of Day filter (e.g., America/Chicago)."
)
custom_tz_str = ""
target_tz_str = "" # Initialize
if selected_tz_option == "Custom":
    custom_tz_str = st.sidebar.text_input("Enter Custom Timezone (IANA Format)", placeholder="e.g., Europe/Paris")
    target_tz_str = custom_tz_str
else:
    target_tz_str = tz_shortcuts.get(selected_tz_option)

# Day of Week Filter
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_days = st.sidebar.multiselect(
    "Select Day(s) of Week",
    options=days,
    default=["Tuesday", "Wednesday", "Thursday"] # Example default
)

# Hour of Day Filter (Start of Initial Range)
# Assuming 1-hour initial range, 1-hour post-range window
selected_hour = st.sidebar.slider(
    "Select Initial Range Start Hour",
    min_value=0,
    max_value=23,
    value=8, # Example: 8 AM (meaning 08:00-09:00 initial range)
    format="%d:00",
    help="Start hour (0-23) for the 1-hour initial range (e.g., 8 means 08:00-09:00). RTO is checked in the hour immediately following (09:00-10:00)."
)

# Previous Hour State Filter
prev_state_options = ["All", "Bullish", "Bearish"]
selected_prev_state = st.sidebar.selectbox(
    "Filter by Previous Hour State?",
    options=prev_state_options,
    index=0, # Default to All
    help="Filter based on whether the hour *before* the initial range was Bullish (Close>Open) or Bearish (Close<Open)."
)

# Analysis Button
run_button = st.sidebar.button("📊 Calculate Probability", use_container_width=True)

# --- Main Area for Status and Results ---
st.divider()

if run_button:
    # --- Input Validation ---
    st.subheader("Running Analysis...")
    valid_input = True
    user_timezone = None
    error_messages = []

    if not target_tz_str:
         error_messages.append("❌ Please select or enter a valid timezone.")
         valid_input = False
    else:
         try:
             user_timezone = ZoneInfo(target_tz_str)
         except ZoneInfoNotFoundError:
             error_messages.append(f"❌ Timezone '{target_tz_str}' not found. Use IANA format.")
             valid_input = False
         except Exception as e:
             error_messages.append(f"❌ Error validating timezone '{target_tz_str}': {e}")
             valid_input = False

    if not selected_days:
        error_messages.append("❌ Please select at least one Day of Week.")
        valid_input = False

    if not valid_input:
        for msg in error_messages: st.error(msg)
    else:
        # --- Inputs valid, proceed ---
        master_df_utc = None
        analysis_completed = False
        probability = None
        total_scenarios = 0
        rto_scenarios = 0

        try:
            # 1. Load Data (already UTC)
            data_url = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv'
            with st.spinner('Loading and preparing data... (cached)'):
                master_df_utc = load_data(data_url)

            if master_df_utc is None or master_df_utc.empty:
                raise ValueError("Data loading failed or resulted in empty DataFrame.")

            # 2. Timezone Conversion
            st.write(f"Converting data to target timezone: {user_timezone.key}...")
            df_localized = master_df_utc.tz_convert(user_timezone)

            # Handle potential NaTs from conversion (e.g., DST)
            original_len = len(df_localized)
            df_localized.dropna(inplace=True) # Drop rows where index became NaT
            dropped_rows = original_len - len(df_localized)
            if dropped_rows > 0:
                st.warning(f"⚠️ Dropped {dropped_rows} rows with invalid timestamps (NaT) during timezone conversion (likely DST).")

            if df_localized.empty:
                st.warning(f"⚠️ No data remaining after timezone conversion and NaT removal.")
                raise ValueError("No data after TZ conversion")

            # 3. Run Core Probability Calculation
            st.write("Calculating RTO probability based on filters...")
            with st.spinner("Analyzing historical data..."):
                probability, total_scenarios, rto_scenarios = calculate_rto_probability(
                    df_localized,
                    selected_days,
                    selected_hour,
                    selected_prev_state,
                    user_timezone
                )
            analysis_completed = True
            st.success("✅ Probability calculation complete!")

        except (ConnectionError, ValueError) as data_err:
             st.error(f"❌ Data Error: {data_err}")
        except Exception as e:
             st.error(f"❌ An unexpected error occurred during analysis: {e}")
             st.exception(e) # Show full traceback for debugging

        # --- Display Results ---
        st.divider()
        if analysis_completed:
            st.header("📊 Probability Results")

            # Display Parameters Used
            st.markdown("**Filters Applied:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Timezone: {user_timezone.key}")
                st.write(f"Day(s): {', '.join(selected_days)}")
            with col2:
                st.write(f"Initial Range Start: {selected_hour:02d}:00")
                st.write(f"Previous Hour State: {selected_prev_state}")
            st.markdown("---")


            if probability is None:
                st.warning("Could not calculate probability. No valid breakout scenarios found for the selected filters.")
            else:
                st.metric(
                    label="Probability of Post-Breakout RTO",
                    value=f"{probability:.2f}%",
                    delta=f"{rto_scenarios} RTOs / {total_scenarios} Breakouts",
                    delta_color="off"
                )

                # --- Create Plotly Gauge Chart ---
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"P(RTO | Breakout)", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "cornflowerblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 25], 'color': 'rgba(255, 0, 0, 0.3)'},    # Red zone (Low %)
                            {'range': [25, 50], 'color': 'rgba(255, 165, 0, 0.3)'}, # Orange zone
                            {'range': [50, 75], 'color': 'rgba(255, 255, 0, 0.3)'}, # Yellow zone
                            {'range': [75, 100], 'color': 'rgba(0, 128, 0, 0.3)'}  # Green zone (High %)
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90 # Example threshold line
                        }
                    }
                ))

                fig.update_layout(
                    # paper_bgcolor = "lightsteelblue",
                    font = {'color': "darkblue", 'family': "Arial"},
                    height=350 # Adjust height as needed
                 )

                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Based on {total_scenarios} historical breakout scenarios matching the filters.")

        elif not valid_input:
             pass # Errors already shown
        else:
             st.warning("Analysis did not complete successfully. Check parameters and data validity.")

else:
    st.info("ℹ️ Configure filters in the sidebar and click 'Calculate Probability'.")
