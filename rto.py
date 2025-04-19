# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import datetime, date, time as dt_time, timedelta
import math # Added for probability calculation
from decimal import Decimal # <-- Add this import

# --- Configuration ---
OANDA_API_BASE_URL = "https://api-fxpractice.oanda.com/v3"
DEFAULT_INSTRUMENT = "NAS100_USD"
DEFAULT_GRANULARITY = "M1"
DEFAULT_COUNT = 180
TARGET_TIMEZONE_STR = "America/Chicago"
HISTORICAL_DATA_URL = 'https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv' # Added for probability

# =============================================================================
# Data Loading Function (Historical Data for Probability)
# =============================================================================
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_historical_data(url):
    """Loads historical data from URL, prepares OHLCV columns, and returns DataFrame (UTC)."""
    try:
        # Use st.spinner for better feedback during potentially long loads
        with st.spinner(f"Loading historical data from {url}... (cached)"):
            temp_df = pd.read_csv(url)

            # Basic Column Validation
            if 'ts_event' not in temp_df.columns or 'symbol' not in temp_df.columns:
                raise ValueError("Essential columns 'ts_event' or 'symbol' missing in historical data.")

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
                raise ValueError(f"Missing required OHLCV columns in historical data: {missing}")

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
                st.sidebar.warning(f"Hist. Data Warning: Dropped {nan_rows} row(s) with NaNs.")

            if master_df.empty:
                 st.sidebar.error("Historical data loaded, but became empty after cleaning.")
                 return None # Return None to indicate failure

        st.info("Historical data loaded.")
        return master_df

    except FileNotFoundError:
        st.sidebar.error(f"Hist. Data Error: Could not find file at URL: {url}")
    except pd.errors.EmptyDataError:
        st.sidebar.error(f"Hist. Data Error: The file at {url} is empty.")
    except Exception as e:
        st.sidebar.error(f"Hist. Data Error: {e}")
    return None # Return None on error


# --- Helper Function to Fetch OANDA Data ---
def fetch_oanda_data(api_key, account_id, instrument, granularity, count, start_time_utc_str=None):
    """Fetches candle data from OANDA API and converts timezone."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{OANDA_API_BASE_URL}/instruments/{instrument}/candles"
    response_obj = None
    params = {"granularity": granularity, "price": "M"} # Midpoint price

    # Adjust parameters based on whether a start time is provided
    if start_time_utc_str:
        params["from"] = start_time_utc_str
        params["count"] = count # OANDA fetches *up to* count from 'from' time
        st.sidebar.caption(f"Fetching {count} candles from {start_time_utc_str[:10]} ...")
    else:
        params["count"] = count # Fetch the latest 'count' candles
        st.sidebar.caption(f"Fetching latest {count} candles")

    try:
        response_obj = requests.get(url, headers=headers, params=params)
        response_obj.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response_obj.json()

        if 'candles' not in data or not data['candles']:
            st.warning(f"No candle data received for {instrument} ({granularity}). Check parameters or API status.")
            return None

        records = []
        for candle in data['candles']:
            # Only process complete candles
            if not candle['complete']:
                continue
            time_utc = pd.to_datetime(candle['time'])
            volume = candle['volume']
            # Extract midpoint prices
            open_price = float(candle['mid']['o'])
            high_price = float(candle['mid']['h'])
            low_price = float(candle['mid']['l'])
            close_price = float(candle['mid']['c'])
            records.append({
                'time': time_utc,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        if not records:
            st.warning("All candles received were incomplete. No data to display.")
            return None

        df = pd.DataFrame(records)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True) # Ensure chronological order

        # Convert timezone from UTC (OANDA default) to target timezone
        try:
            target_tz_obj = ZoneInfo(TARGET_TIMEZONE_STR)
            df = df.tz_convert(target_tz_obj)
        except ZoneInfoNotFoundError:
            st.error(f"Timezone '{TARGET_TIMEZONE_STR}' not found. Using UTC.")
        except Exception as tz_error:
            st.error(f"Timezone conversion error: {tz_error}. Using UTC.")
            # Optionally, keep df as UTC: df = df.tz_localize(None).tz_localize('UTC')

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"OANDA API request failed: {e}")
        if response_obj is not None:
            st.error(f"Response Status Code: {response_obj.status_code}")
            st.error(f"Response Text: {response_obj.text}")
    except Exception as e:
        st.error(f"Error processing OANDA data: {e}")

    return None


# =============================================================================
# Core Probability Calculation Function (Adapted for specific scenario)
# =============================================================================
def calculate_rto_probability_for_scenario(df_hist_localized, target_day_name, initial_range_start_hour, prev_hour_state_filter, user_timezone):
    """
    Calculates the probability of RTO for a specific historical scenario.

    Args:
        df_hist_localized (pd.DataFrame): The historical DataFrame already localized to user_timezone.
        target_day_name (str): Specific day name (e.g., "Monday").
        initial_range_start_hour (int): The hour (0-23) when the initial range starts.
        prev_hour_state_filter (str): "Bullish", "Bearish", or "All".
        user_timezone (ZoneInfo): The timezone object used for localization.

    Returns:
        tuple: (probability, total_breakout_scenarios, rto_occurred_scenarios)
               Returns (None, 0, 0) if no valid scenarios found or on error.
    """
    total_breakout_scenarios = 0
    rto_occurred_scenarios = 0

    if df_hist_localized is None or df_hist_localized.empty:
        # st.sidebar.warning("Historical DataFrame empty. Cannot calculate probability.") # Avoid excessive warnings
        return None, 0, 0

    # 1. Filter by the specific Day of Week
    df_filtered_day = df_hist_localized[df_hist_localized.index.day_name() == target_day_name]

    if df_filtered_day.empty:
        # st.sidebar.warning(f"No historical data found for {target_day_name}.") # Avoid excessive warnings
        return None, 0, 0

    unique_days_in_data = df_filtered_day.index.normalize().unique()
    if unique_days_in_data.empty:
        return None, 0, 0

    # --- Define Time Windows based on input hour ---
    try:
        initial_range_start_t = dt_time(initial_range_start_hour, 0)
        # Assuming 1-hour initial range, 1-hour post-range window
        initial_range_end_t = dt_time((initial_range_start_hour + 1) % 24, 0)
        post_range_end_t = dt_time((initial_range_start_hour + 2) % 24, 0)
        prev_hour_start_t = dt_time((initial_range_start_hour - 1 + 24) % 24, 0) # Handle wrap-around
        prev_hour_end_t = initial_range_start_t # End of prev hour is start of initial range

        # Handle overnight ranges
        initial_range_crosses_midnight = initial_range_end_t < initial_range_start_t
        post_range_crosses_midnight = post_range_end_t < initial_range_end_t
        prev_hour_crosses_midnight = prev_hour_end_t < prev_hour_start_t

    except Exception as e:
        st.sidebar.error(f"Prob Calc Error (Time Windows): {e}")
        return None, 0, 0

    # --- Process each unique day matching the target_day_name ---
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
            if prev_hour_data.empty: continue

            try:
                prev_open = prev_hour_data['Open'].iloc[0]
                prev_close = prev_hour_data['Close'].iloc[-1]
                is_bullish_prev = prev_close > prev_open
                is_bearish_prev = prev_close < prev_open
            except IndexError: continue

            # Apply Previous Hour State Filter
            if prev_hour_state_filter == "Bullish" and not is_bullish_prev: continue
            if prev_hour_state_filter == "Bearish" and not is_bearish_prev: continue
            # "All" passes through if prev_hour_state_filter is "All"

            # --- 2. Analyze Initial Range ---
            initial_range_data = df_filtered_day.loc[initial_range_start_dt:initial_range_end_exclusive_dt]
            if initial_range_data.empty: continue

            range_high = initial_range_data['High'].max()
            range_low = initial_range_data['Low'].min()

            # --- 3. Get Reference Price & Post-Range Data ---
            post_range_data = df_filtered_day.loc[post_range_start_dt:post_range_end_exclusive_dt_day]
            if post_range_data.empty: continue

            try:
                reference_open_price = post_range_data['Open'].iloc[0]
            except IndexError: continue

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
            total_breakout_scenarios += 1 # Increment total scenarios *only if* a breakout happened under the filtered conditions

            rto_check_data = post_range_data[post_range_data.index > first_breakout_time]
            if not rto_check_data.empty:
                post_rto_condition = (rto_check_data['Low'] <= reference_open_price) & (rto_check_data['High'] >= reference_open_price)
                if post_rto_condition.any():
                    rto_occurred_scenarios += 1

        except Exception as e:
            # st.sidebar.warning(f"Skipping day {day_date.date()} due to error: {e}") # Avoid excessive warnings
            continue

    # --- Calculate Final Probability ---
    if total_breakout_scenarios == 0:
        return None, 0, 0
    else:
        probability = (rto_occurred_scenarios / total_breakout_scenarios) * 100
        return probability, total_breakout_scenarios, rto_occurred_scenarios


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("📈 Breakout & RTO Analysis")

# --- Sidebar for Inputs ---
st.sidebar.header("OANDA Credentials")
api_key = st.sidebar.text_input("API Key", value="", type="password", help="Your OANDA Practice API Key.")
account_id = st.sidebar.text_input("Account ID", value="", help="Your OANDA Practice Account ID.")
st.sidebar.markdown("---")
st.sidebar.header("Chart Parameters")
instrument = st.sidebar.text_input("Instrument", value=DEFAULT_INSTRUMENT)
granularity = st.sidebar.selectbox(
    "Timeframe",
    ['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'],
    index=4 # Default to M1
)
count = st.sidebar.number_input("Candles", min_value=10, max_value=5000, value=DEFAULT_COUNT, step=10, help="Number of candles to fetch.")
st.sidebar.markdown("---")
use_custom_start = st.sidebar.checkbox("Custom Start Time", value=False, help="Fetch data starting from a specific date/time instead of the latest.")
start_time_api_str = None # Initialize
target_tz = None # Initialize target timezone object

# Get target timezone object safely
try:
    target_tz = ZoneInfo(TARGET_TIMEZONE_STR)
except ZoneInfoNotFoundError:
    st.sidebar.error(f"Default timezone '{TARGET_TIMEZONE_STR}' not found. Using UTC.")
    TARGET_TIMEZONE_STR = "UTC" # Fallback
    target_tz = ZoneInfo("UTC")
except Exception as e:
    st.sidebar.error(f"Timezone loading error: {e}. Using UTC.")
    TARGET_TIMEZONE_STR = "UTC" # Fallback
    target_tz = ZoneInfo("UTC")


if use_custom_start:
    now_local = datetime.now(target_tz) # Use the validated target_tz
    start_date_input = st.sidebar.date_input("Start Date", value=now_local.date())
    start_time_input = st.sidebar.time_input(
        f"Start Time ({TARGET_TIMEZONE_STR})",
        value=dt_time(0, 0), # Default to midnight
        step=timedelta(hours=1) # Step by hour
    )
    if start_date_input and start_time_input:
        try:
            # Combine date and time, make it timezone-aware in the target timezone
            start_dt_naive = datetime.combine(start_date_input, start_time_input)
            # Correct way using zoneinfo:
            start_dt_aware_target = start_dt_naive.replace(tzinfo=target_tz)

            # Convert to UTC for the API call
            start_dt_utc = start_dt_aware_target.astimezone(ZoneInfo("UTC"))

            # Format for OANDA API (RFC3339)
            start_time_api_str = start_dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        except Exception as e:
            st.sidebar.error(f"Start time processing error: {e}")
            start_time_api_str = None # Reset on error

# --- Main Logic ---
if st.sidebar.button("Load Chart"):
    # Input Validation
    valid_inputs = True
    if not api_key:
        st.warning("⚠️ OANDA API Key is required.")
        valid_inputs = False
    if not account_id:
        st.warning("⚠️ OANDA Account ID is required.")
        valid_inputs = False
    if not instrument:
        st.warning("⚠️ Instrument is required.")
        valid_inputs = False

    if valid_inputs:
        st.info(f"Requesting data for {instrument} ({granularity}) from OANDA API...")
        # Fetch OANDA data using the helper function
        df = fetch_oanda_data(api_key, account_id, instrument, granularity, count, start_time_api_str)

        # --- Load Historical Data (will use cache if available) ---
        df_historical_utc = load_historical_data(HISTORICAL_DATA_URL)
        df_historical_localized = None
        # target_tz_obj already defined as target_tz
        try:
            if df_historical_utc is not None:
                 df_historical_localized = df_historical_utc.tz_convert(target_tz)
                 # Optional: Handle NaTs after conversion if needed, though less critical for stats
                 # df_historical_localized.dropna(inplace=True)
        except Exception as e:
            st.sidebar.error(f"Error converting historical data timezone: {e}")
        # --- End Historical Data Load ---


        # Proceed only if data fetching was successful
        if df is not None and not df.empty:
            # --- Display Latest Date Info ---
            latest_timestamp = None # Initialize
            day_of_week = None
            try:
                latest_timestamp = df.index[-1]
                day_of_week = latest_timestamp.strftime('%A') # Get current day name
                date_str = latest_timestamp.strftime('%B %d, %Y')
                st.sidebar.markdown("---")
                st.sidebar.info(f"{day_of_week} {date_str}")
            except Exception: pass # Ignore if error getting latest date

            # --- Create Plotly Chart ---
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                open=df['open'],
                                                high=df['high'],
                                                low=df['low'],
                                                close=df['close'],
                                                name=instrument,
                                                increasing=dict(line=dict(color='royalblue'), fillcolor='royalblue'),
                                                decreasing=dict(line=dict(color='lightgrey'), fillcolor='lightgrey'))])

            # Initialize variables for analysis
            hour_low, hour_high, last_completed_hour_end = None, None, None
            last_completed_hour_start = None # Initialize this
            hour_status = "Neutral" # Initialize default status
            first_candle_high_match_x, first_candle_high_match_y = [], []
            first_candle_low_match_x, first_candle_low_match_y = [], []
            latest_hr_high_match_x, latest_hr_high_match_y = [], []
            latest_hr_low_match_x, latest_hr_low_match_y = [], []
            latest_hour_open_price = None
            first_candle_latest_time = None # Timestamp of the first candle of the latest hour

            # --- Calculate Latest Hour Open Price & First Candle Time ---
            try:
                if not df.empty:
                    latest_time_line = df.index[-1]
                    latest_hour_start = latest_time_line.floor('h')
                    # Get all candles from the start of the latest hour onwards
                    first_candle_series = df.loc[df.index >= latest_hour_start]
                    if not first_candle_series.empty:
                        latest_hour_open_price = first_candle_series.iloc[0]['open']
                        first_candle_latest_time = first_candle_series.iloc[0].name # Get timestamp of the first candle
            except Exception as e:
                st.sidebar.error(f"Could not calculate latest hour info: {e}")


            # --- Hourly Box Logic (Last Completed Hour) & First Candle Check ---
            try:
                if len(df.index) > 1: # Need at least two candles to define a previous hour
                    latest_time_box = df.index[-1]
                    current_hour_start_box = latest_time_box.floor('h')
                    last_completed_hour_start = current_hour_start_box - timedelta(hours=1)
                    last_completed_hour_end = current_hour_start_box # End is exclusive for loc

                    # Filter data for the last completed hour
                    df_last_hour = df.loc[(df.index >= last_completed_hour_start) & (df.index < last_completed_hour_end)]

                    if not df_last_hour.empty:
                        hour_low = df_last_hour['low'].min()
                        hour_high = df_last_hour['high'].max()
                        first_candle_of_hour = df_last_hour.iloc[0]
                        last_candle_of_hour = df_last_hour.iloc[-1]
                        hour_open = first_candle_of_hour['open']
                        hour_close = last_candle_of_hour['close']

                        # Determine hour status and box color
                        # hour_status = "Neutral" # Already initialized
                        box_border_color = "Yellow"
                        if hour_close > hour_open:
                            hour_status = "Bullish"
                            box_border_color = "lightgreen"
                        elif hour_close < hour_open:
                            hour_status = "Bearish"
                            box_border_color = "lightcoral"

                        # Add the rectangle shape for the hour range
                        fig.add_shape(type="rect",
                                      x0=last_completed_hour_start, x1=last_completed_hour_end,
                                      y0=hour_low, y1=hour_high,
                                      line=dict(color=box_border_color, width=1),
                                      fillcolor="rgba(255, 255, 0, 0.10)", # Semi-transparent yellow fill
                                      layer="below") # Draw below candles

                        # Check if first candle's high/low matches the hour's high/low
                        first_candle_time = first_candle_of_hour.name
                        first_candle_high = first_candle_of_hour['high']
                        first_candle_low = first_candle_of_hour['low']
                        match_info = []
                        if first_candle_high == hour_high:
                            first_candle_high_match_x.append(first_candle_time)
                            first_candle_high_match_y.append(first_candle_high)
                            match_info.append("Imb High")
                        if first_candle_low == hour_low:
                            first_candle_low_match_x.append(first_candle_time)
                            first_candle_low_match_y.append(first_candle_low)
                            match_info.append("Imb Low")

                        # Sidebar info moved below probability calc

            except Exception as e:
                st.sidebar.error(f"Error processing hourly box/state: {e}")


            # --- Calculate and Display RTO Probability ---
            st.sidebar.markdown("---") # Separator
            st.sidebar.subheader("Historical RTO Probability")
            if df_historical_localized is not None and target_tz is not None and day_of_week is not None and last_completed_hour_start is not None:
                prob_calc_hour = last_completed_hour_start.hour
                # Map Neutral state from OANDA chart to "All" for probability function
                prob_calc_prev_state = hour_status if hour_status != "Neutral" else "All"

                probability, total_scenarios, rto_scenarios = calculate_rto_probability_for_scenario(
                    df_historical_localized,
                    day_of_week,
                    prob_calc_hour,
                    prob_calc_prev_state,
                    target_tz # Use the validated target_tz object
                )

                if probability is not None:
                    st.sidebar.metric(
                        label="P(RTO | Breakout)",
                        value=f"{probability:.1f}%",
                        delta=f"{rto_scenarios}/{total_scenarios} scenarios",
                        delta_color="off"
                    )
                elif total_scenarios == 0:
                     st.sidebar.warning("No matching historical scenarios found.")
                else: # Error occurred during calculation
                     st.sidebar.error("Could not calculate probability.")

            else:
                missing_info = []
                if df_historical_localized is None: missing_info.append("Hist. Data")
                if target_tz is None: missing_info.append("Timezone")
                if day_of_week is None: missing_info.append("Current Day")
                if last_completed_hour_start is None: missing_info.append("Prev. Hour")
                st.sidebar.warning(f"Cannot calculate probability (Missing: {', '.join(missing_info)})")
            st.sidebar.markdown("---") # Separator
            # --- End Probability Calculation ---


            # --- Display Previous Hour Info (Moved here) ---
            if last_completed_hour_start and last_completed_hour_end: # Check both exist
                 st.sidebar.info(f"Prev Hr State: {hour_status}")
                 st.sidebar.info(f"Time: {last_completed_hour_start.strftime('%H:%M')}-{last_completed_hour_end.strftime('%H:%M')} {TARGET_TIMEZONE_STR.split('/')[-1]}")
                 if hour_low is not None and hour_high is not None:
                     st.sidebar.info(f"Range: {hour_low:.2f}-{hour_high:.2f}")
                 # Display Imbalance info if calculated (using match_info from box logic)
                 if 'match_info' in locals() and match_info: # Check if match_info was defined
                     st.sidebar.info(f"Prev Hr Info: {'; '.join(match_info)}")
            # --- End Previous Hour Info ---


            # --- Markers for First Candle H/L Match (Completed Hour) ---
            if first_candle_high_match_x:
                fig.add_trace(go.Scatter(x=first_candle_high_match_x, y=first_candle_high_match_y, mode='markers',
                                         marker=dict(symbol='star', color='lightgreen', size=10),
                                         name='Prev Hr Imb High', showlegend=True,
                                         hovertemplate = 'Prev Hr Imb High<extra></extra>'))
            if first_candle_low_match_x:
                fig.add_trace(go.Scatter(x=first_candle_low_match_x, y=first_candle_low_match_y, mode='markers',
                                         marker=dict(symbol='star', color='lightcoral', size=10),
                                         name='Prev Hr Imb Low', showlegend=True,
                                         hovertemplate = 'Prev Hr Imb Low<extra></extra>'))

            # --- Check First Candle of Latest Hour vs Its Range So Far ---
            try:
                 if not df.empty and first_candle_latest_time is not None: # Need first candle time here
                    latest_hour_start = first_candle_latest_time.floor('h') # Use actual first candle time's hour start
                    df_latest_hour = df.loc[df.index >= latest_hour_start] # Filter from actual start

                    if not df_latest_hour.empty and len(df_latest_hour) > 0: # Check if there's data in the latest hour
                        latest_hour_high = df_latest_hour['high'].max()
                        latest_hour_low = df_latest_hour['low'].min()

                        # Get the first candle data using its known timestamp index
                        first_candle_latest = df_latest_hour.loc[first_candle_latest_time]
                        first_candle_latest_high = first_candle_latest['high']
                        first_candle_latest_low = first_candle_latest['low']

                        latest_match_info = []
                        if first_candle_latest_high == latest_hour_high:
                            latest_hr_high_match_x.append(first_candle_latest_time)
                            latest_hr_high_match_y.append(first_candle_latest_high)
                            latest_match_info.append("High")
                        if first_candle_latest_low == latest_hour_low:
                            latest_hr_low_match_x.append(first_candle_latest_time)
                            latest_hr_low_match_y.append(first_candle_latest_low)
                            latest_match_info.append("Low")

                        if latest_match_info:
                            st.sidebar.info(f"Last Hr Info: Imb {'; '.join(latest_match_info)}")
            except KeyError:
                 st.sidebar.warning("Could not find first candle of latest hour by timestamp (KeyError).") # Specific error
            except Exception as e:
                 st.sidebar.error(f"Error checking latest hour 1st candle: {e}")


            # --- Markers for First Candle H/L Match (Latest Hour) ---
            if latest_hr_high_match_x:
                fig.add_trace(go.Scatter(x=latest_hr_high_match_x, y=latest_hr_high_match_y, mode='markers',
                                         marker=dict(symbol='circle', color='lightblue', size=8, line=dict(color='white', width=1)),
                                         name='Last Hr Imb High', showlegend=True,
                                         hovertemplate = 'Last Hr Imb High<extra></extra>'))
            if latest_hr_low_match_x:
                fig.add_trace(go.Scatter(x=latest_hr_low_match_x, y=latest_hr_low_match_y, mode='markers',
                                         marker=dict(symbol='circle', color='purple', size=8, line=dict(color='white', width=1)),
                                         name='Last Hr Imb Low', showlegend=True,
                                         hovertemplate = 'Last Hr Imb Low<extra></extra>'))

            # --- Highlight 32nd Minute Candle (if M1 timeframe) ---
            try:
                if granularity == 'M1' and not df.empty:
                    latest_time_m32 = df.index[-1]
                    latest_hour_start_m32 = latest_time_m32.floor('h')
                    target_time_m32 = latest_hour_start_m32 + timedelta(minutes=32)

                    # Check if the target time exists in the DataFrame index
                    if target_time_m32 in df.index:
                        target_candle_data = df.loc[target_time_m32]
                        # Place marker slightly above the high for visibility
                        marker_y_pos = target_candle_data['high'] * 1.0005 # Adjust multiplier as needed
                        fig.add_trace(go.Scatter(x=[target_time_m32], y=[marker_y_pos], mode='markers',
                                                 marker=dict(symbol='circle', color='red', size=7),
                                                 name='32m Mark', showlegend=False, # Usually don't need legend for this
                                                 hovertemplate = f'32nd min ({target_time_m32.strftime("%H:%M")})<extra></extra>'))
            except Exception as e:
                st.sidebar.error(f"Error marking 32m candle: {e}")


            # --- Breakout & Return-to-Open Logic (Refined Version) ---
            pre_rto_marker_x, pre_rto_marker_y = [], []
            post_rto_marker_x, post_rto_marker_y = [], []
            first_breakout_time = None
            is_bullish_breakout = None # None: No breakout, True: Bullish, False: Bearish
            pre_rto_occurred = False
            post_rto_occurred = False
            first_post_rto_time = None
            first_pre_rto_time = None # Track time of first pre-rto

            # Ensure we have the necessary components
            if hour_high is not None and hour_low is not None and last_completed_hour_end is not None and latest_hour_open_price is not None:
                try:
                    # Data after the 'initial range' (previous completed hour)
                    df_after_hour = df.loc[df.index >= last_completed_hour_end]

                    if not df_after_hour.empty:

                        # --- 1. Find First Breakout & Direction ---
                        high_break_condition = df_after_hour['high'] > hour_high
                        first_high_break_time = df_after_hour[high_break_condition].index.min() if high_break_condition.any() else None
                        low_break_condition = df_after_hour['low'] < hour_low
                        first_low_break_time = df_after_hour[low_break_condition].index.min() if low_break_condition.any() else None

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

                        # --- 2. Check for Pre-RTO and Post-RTO ---
                        if first_breakout_time is not None:
                            # --- 2a. Check for Pre-Breakout RTO ---
                            # Data between range end and breakout time (exclusive of breakout bar)
                            pre_breakout_check_data = df_after_hour[df_after_hour.index < first_breakout_time]
                            if not pre_breakout_check_data.empty:
                                # Check if any candle in this period touches the reference open price
                                pre_rto_condition = (pre_breakout_check_data['low'] <= latest_hour_open_price) & (pre_breakout_check_data['high'] >= latest_hour_open_price)
                                if pre_rto_condition.any():
                                    pre_rto_occurred = True
                                    # Find the time of the *first* pre-RTO event
                                    first_pre_rto_time = pre_breakout_check_data[pre_rto_condition].index.min()
                                    if first_pre_rto_time:
                                        pre_rto_marker_x.append(first_pre_rto_time)
                                        pre_rto_marker_y.append(latest_hour_open_price)

                            # --- 2b. Check for Post-Breakout RTO ---
                            # Data strictly *after* the breakout candle
                            post_breakout_check_data = df_after_hour[df_after_hour.index > first_breakout_time]
                            if not post_breakout_check_data.empty:
                                post_rto_condition = (post_breakout_check_data['low'] <= latest_hour_open_price) & (post_breakout_check_data['high'] >= latest_hour_open_price)
                                if post_rto_condition.any():
                                    post_rto_occurred = True
                                    # Find the time of the *first* post-RTO event
                                    first_post_rto_time = post_breakout_check_data[post_rto_condition].index.min()
                                    if first_post_rto_time:
                                        post_rto_marker_x.append(first_post_rto_time)
                                        post_rto_marker_y.append(latest_hour_open_price)

                        # --- 3. Add Markers and Sidebar Info ---
                        if is_bullish_breakout is True:
                            upside_bo_high = df.loc[first_breakout_time, 'high']
                            fig.add_trace(go.Scatter(x=[first_breakout_time], y=[upside_bo_high], mode='markers',
                                                     marker=dict(symbol='triangle-up', color='lime', size=10),
                                                     name='Bullish Breakout', showlegend=True,
                                                     hovertemplate = f'Bullish BO @ {first_breakout_time.strftime("%H:%M")}<extra></extra>'))
                            st.sidebar.info(f"Bullish BO at {first_breakout_time.strftime('%H:%M')} {TARGET_TIMEZONE_STR.split('/')[-1]}")
                        elif is_bullish_breakout is False:
                            downside_bo_low = df.loc[first_breakout_time, 'low']
                            fig.add_trace(go.Scatter(x=[first_breakout_time], y=[downside_bo_low], mode='markers',
                                                     marker=dict(symbol='triangle-down', color='red', size=10),
                                                     name='Bearish Breakout', showlegend=True,
                                                     hovertemplate = f'Bearish BO @ {first_breakout_time.strftime("%H:%M")}<extra></extra>'))
                            st.sidebar.info(f"Bearish BO at {first_breakout_time.strftime('%H:%M')} {TARGET_TIMEZONE_STR.split('/')[-1]}")
                        else: # No breakout occurred in the window
                             st.sidebar.info("No breakout yet in current hour.")

                        # Add Pre-RTO marker if it occurred
                        if pre_rto_marker_x:
                            fig.add_trace(go.Scatter(x=pre_rto_marker_x, y=pre_rto_marker_y, mode='markers',
                                                     marker=dict(symbol='circle-open', color='cyan', size=9, line=dict(width=2)),
                                                     name='Pre-RTO', showlegend=True,
                                                     hovertemplate = f'Pre-RTO @ {first_pre_rto_time.strftime("%H:%M")}<extra></extra>'))
                            st.sidebar.info(f"Pre-RTO at {first_pre_rto_time.strftime('%H:%M')} {TARGET_TIMEZONE_STR.split('/')[-1]}")
                        elif first_breakout_time: # Only mention if breakout happened
                            st.sidebar.info("No Pre-RTO detected.")

                        # Add Post-RTO marker if it occurred
                        if post_rto_marker_x:
                            fig.add_trace(go.Scatter(x=post_rto_marker_x, y=post_rto_marker_y, mode='markers',
                                                     marker=dict(symbol='diamond', color='magenta', size=10, line=dict(color='white', width=1)),
                                                     name='Post-RTO', showlegend=True,
                                                     hovertemplate = f'Post-RTO @ {first_post_rto_time.strftime("%H:%M")}<extra></extra>'))
                            st.sidebar.info(f"Post-RTO at {first_post_rto_time.strftime('%H:%M')} {TARGET_TIMEZONE_STR.split('/')[-1]}")
                        elif first_breakout_time: # Only mention if breakout happened
                            st.sidebar.info("No Post-RTO detected yet.")

                    else: # df_after_hour is empty
                        st.sidebar.warning("No data found after the previous completed hour.")

                except Exception as e:
                    st.sidebar.error(f"Error checking breakouts/RTO: {e}")
                    st.exception(e) # Show full traceback in console/log for debugging
            elif latest_hour_open_price is None:
                st.sidebar.warning("Cannot check RTO: Latest hour open price missing.")
            elif hour_high is None or hour_low is None:
                 st.sidebar.warning("Cannot check Breakout/RTO: Previous hour range missing.")
            # --- End of Breakout & RTO Logic ---


            # --- Hourly Open Line Segment Logic ---
            try:
                # Ensure we have the open price and the timestamp of the first candle
                if latest_hour_open_price is not None and first_candle_latest_time is not None and not df.empty:
                     start_time_of_line = first_candle_latest_time
                     end_time_of_line = df.index[-1] # Extend line to the last available candle

                     # Only draw if the start and end times are different (more than one candle in the hour)
                     if start_time_of_line != end_time_of_line:
                         fig.add_shape(type="line",
                                       x0=start_time_of_line, y0=latest_hour_open_price,
                                       x1=end_time_of_line, y1=latest_hour_open_price,
                                       line=dict(color="yellow", width=1, dash="dash"),
                                       layer="below") # Draw below candles

                       # --- Corrected Line Below ---
                         # Add annotation for the open price near the end of the line
                         fig.add_annotation(x=end_time_of_line, y=latest_hour_open_price,
                                            xref="x", yref="y",
                                            # Use decimal.Decimal instead of pd.Decimal
                                            text=f"Open: {latest_hour_open_price:.{df['open'].apply(lambda x: abs(Decimal(str(x)).as_tuple().exponent)).max()}f}", # Dynamic precision
                                            showarrow=False,
                                            yshift=5, # Shift text slightly above the line
                                            xanchor="right", # Anchor text to the right
                                            font=dict(size=10, color="yellow"))
                         # --- End Corrected Line ---

                         st.sidebar.info(f"Open at {start_time_of_line.strftime('%H:%M')} {TARGET_TIMEZONE_STR.split('/')[-1]}: {latest_hour_open_price:.2f}")
            except Exception as e:
                st.sidebar.error(f"Error drawing line segment: {e}")

            # --- Final Layout Update ---
            fig.update_layout(
                title=f'{instrument} ({granularity}) Chart',
                xaxis_title=f'Time ({TARGET_TIMEZONE_STR})',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False, # Hide range slider
                template='plotly_dark', # Use dark theme
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), # Legend top-right horizontal
                # width=1000, # Adjust width as needed, or let Streamlit manage
                height=800, # Adjust height
                hovermode='x unified', # Show unified hover info for all traces at x-coordinate
                xaxis=dict(
                    title=f'Time ({TARGET_TIMEZONE_STR})',
                    showspikes=True, # Show spike lines on hover
                    spikemode='across', # Spike line across both axes
                    spikesnap='cursor', # Snap spike to cursor
                    spikedash='dot',
                    spikecolor='grey',
                    spikethickness=1
                ),
                yaxis=dict(
                    title='Price',
                    showspikes=True,
                    spikemode='across',
                    spikesnap='cursor',
                    spikedash='dot',
                    spikecolor='grey',
                    spikethickness=1
                )
            )

            # --- Display Chart ---
            st.plotly_chart(fig, use_container_width=True) # Use container width

        else:
            # Error message handled within fetch_oanda_data or if df is None/empty
            st.error("Failed to load or process OANDA data. Check sidebar messages.")
else:
    st.info("ℹ️ Enter OANDA details and chart parameters in the sidebar, then click 'Load Chart'.")
