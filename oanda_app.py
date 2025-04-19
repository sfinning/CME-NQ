import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time # Required for time.sleep if using auto-refresh loop later (not implemented yet)
# Make sure zoneinfo is available (Python 3.9+)
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
# Import datetime class and specific types from datetime module
from datetime import datetime, date, time as dt_time, timedelta

# --- Configuration ---
OANDA_API_BASE_URL = "https://api-fxpractice.oanda.com/v3"
DEFAULT_INSTRUMENT = "NAS100_USD"
DEFAULT_GRANULARITY = "M1"
DEFAULT_COUNT = 180
TARGET_TIMEZONE_STR = "America/Chicago" # Define target timezone string

# --- Helper Function to Fetch OANDA Data ---
# Modified to accept optional start_time_utc_str
def fetch_oanda_data(api_key, account_id, instrument, granularity, count, start_time_utc_str=None):
    """
    Fetches historical candle data from OANDA.
    If start_time_utc_str is provided (RFC3339 format), fetches 'count' candles starting from that time.
    Otherwise, fetches the latest 'count' candles.
    Converts time index to the target timezone using zoneinfo.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{OANDA_API_BASE_URL}/instruments/{instrument}/candles"
    response_obj = None

    # --- Build API Parameters ---
    params = {
        "granularity": granularity,
        "price": "M" # Always use midpoint price candles
    }
    if start_time_utc_str:
        params["from"] = start_time_utc_str
        params["count"] = count # Get 'count' candles starting FROM the start time
        st.sidebar.caption(f"Fetching {count} candles from {start_time_utc_str[:10]}...") # Show truncated time
    else:
        params["count"] = count # Default: Get latest 'count' candles
        st.sidebar.caption(f"Fetching latest {count} candles")
    # --- End Build API Parameters ---

    try:
        response_obj = requests.get(url, headers=headers, params=params)
        response_obj.raise_for_status()
        data = response_obj.json()

        if 'candles' not in data or not data['candles']:
            st.warning(f"No candle data received for {instrument} ({granularity}).")
            return None

        records = []
        for candle in data['candles']:
            if not candle['complete']: continue
            time_utc = pd.to_datetime(candle['time'])
            volume = candle['volume']
            open_price = float(candle['mid']['o']); high_price = float(candle['mid']['h'])
            low_price = float(candle['mid']['l']); close_price = float(candle['mid']['c'])
            records.append({'time': time_utc, 'open': open_price, 'high': high_price,
                            'low': low_price, 'close': close_price, 'volume': volume})

        if not records:
             st.warning("All received candles were incomplete.")
             return None

        df = pd.DataFrame(records)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True) # df index is UTC here

        try:
            target_tz_obj = ZoneInfo(TARGET_TIMEZONE_STR)
            df = df.tz_convert(target_tz_obj) # Convert final df index to target TZ
        except ZoneInfoNotFoundError:
            st.error(f"Timezone '{TARGET_TIMEZONE_STR}' not found.")
            st.error("Proceeding with UTC times.")
        except Exception as tz_error:
            st.error(f"Failed to convert timezone: {tz_error}")
            st.error("Proceeding with UTC times.")

        return df # Return df with index converted to target timezone (or UTC on error)

    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        if response_obj:
             try: st.error(f"OANDA Response: {response_obj.json()}")
             except Exception: st.error(f"OANDA Response (non-JSON): {response_obj.text}")
        return None
    except Exception as e:
        st.error(f"Error fetching/processing data: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("RTO Candlestick Chart")
st.write(f"Connect to OANDA API. Times displayed in {TARGET_TIMEZONE_STR}.")

# --- Sidebar for Inputs ---
st.sidebar.header("OANDA Credentials")
api_key = st.sidebar.text_input("OANDA API Key", value="", type="password", help="Enter your OANDA API access token.")
account_id = st.sidebar.text_input("OANDA Account ID", value="", help="Enter your OANDA Account ID.")
st.sidebar.markdown("---")
st.sidebar.header("Chart Parameters")
instrument = st.sidebar.text_input("Instrument", value=DEFAULT_INSTRUMENT, help="e.g., NAS100_USD, EUR_USD, etc.")
granularity = st.sidebar.selectbox(
    "Timeframe",
    options=['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'h', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'],
    index=4, help="Time interval for candles"
)
count = st.sidebar.number_input(
    "Candles", min_value=10, max_value=5000, value=DEFAULT_COUNT, step=10,
    help="Number of candles to fetch (max 5000)."
)

# --- Custom Start Time Option ---
st.sidebar.markdown("---")
use_custom_start = st.sidebar.checkbox("Use Custom Start Time", value=False)
start_time_api_str = None # Initialize: Fetch latest unless overridden

if use_custom_start:
    # Get today's date in the target timezone for default
    try:
        target_tz = ZoneInfo(TARGET_TIMEZONE_STR)
        # Use datetime.now() with the tz object
        now_local = datetime.now(target_tz)
    except Exception as e:
        st.sidebar.error(f"Could not get local time for {TARGET_TIMEZONE_STR}: {e}")
        now_local = datetime.now() # Fallback to naive system time

    start_date = st.sidebar.date_input("Start Date", value=now_local.date())
    # Default start time to midnight
    start_time = st.sidebar.time_input("Start Time (CT)", value=dt_time(0, 0)) # Label includes TZ

    # Combine date and time, make timezone aware in target timezone, convert to UTC for API
    if start_date and start_time:
        try:
            start_dt_naive = datetime.combine(start_date, start_time)
            # Make aware using the target timezone ZoneInfo object
            start_dt_aware_target = start_dt_naive.replace(tzinfo=target_tz)
            # Convert to UTC
            start_dt_utc = start_dt_aware_target.astimezone(ZoneInfo("UTC"))
            # Format for OANDA API (RFC3339 format, ensure 'Z' for UTC)
            # Example: 2023-04-19T05:00:00.000000Z (microseconds might be optional/truncated by API)
            start_time_api_str = start_dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        except Exception as e:
            st.sidebar.error(f"Error processing start time: {e}")
            start_time_api_str = None # Reset on error
            # Optionally uncheck the box on error: st.session_state['use_custom_start_checkbox_key'] = False (requires setting a key on checkbox)

# --- Main Logic ---
if st.sidebar.button("Load Chart"):
    if not api_key or not account_id or not instrument:
        if not api_key: st.warning("Please enter your OANDA API Key.")
        if not account_id: st.warning("Please enter your OANDA Account ID.")
        if not instrument: st.warning("Please enter an Instrument.")
    else:
        st.info(f"Requesting data for {instrument} ({granularity})...")
        # Pass start_time_api_str (which is None if checkbox is unchecked or error occurred)
        # The fetch function's sidebar caption will indicate the mode used
        df = fetch_oanda_data(api_key, account_id, instrument, granularity, count, start_time_api_str)

        if df is not None and not df.empty:
            # --- PASTE of Charting Logic from previous step ---
            # This block remains the same. It takes the final 'df'
            # (which has the correct data range and timezone)
            # and generates all the visual elements.
            st.success("Data loaded successfully!")
            try:
                if not df.empty:
                    latest_timestamp = df.index[-1]; day_of_week = latest_timestamp.strftime('%A'); date_str = latest_timestamp.strftime('%B %d, %Y')
                    st.sidebar.markdown("---"); st.sidebar.markdown(f"**Latest Data Point ({TARGET_TIMEZONE_STR}):**"); st.sidebar.write(f"Date: {date_str}"); st.sidebar.write(f"Day: {day_of_week}"); st.sidebar.markdown("---")
            except Exception as e: st.sidebar.error(f"Error displaying latest date: {e}")

            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=instrument, increasing=dict(line=dict(color='royalblue'), fillcolor='royalblue'), decreasing=dict(line=dict(color='lightgrey'), fillcolor='lightgrey'))])

            hour_low, hour_high, last_completed_hour_end = None, None, None
            first_candle_high_match_x, first_candle_high_match_y = [], []; first_candle_low_match_x, first_candle_low_match_y = [], []
            latest_hr_high_match_x, latest_hr_high_match_y = [], []; latest_hr_low_match_x, latest_hr_low_match_y = [], []

            try: # Hourly Box Logic
                if len(df.index) > 1:
                    latest_time_box = df.index[-1]; current_hour_start_box = latest_time_box.floor('h')
                    last_completed_hour_start = current_hour_start_box - timedelta(hours=1); last_completed_hour_end = current_hour_start_box
                    df_last_hour = df.loc[(df.index >= last_completed_hour_start) & (df.index < last_completed_hour_end)]
                    if not df_last_hour.empty:
                        hour_low = df_last_hour['low'].min(); hour_high = df_last_hour['high'].max()
                        first_candle_of_hour = df_last_hour.iloc[0]; last_candle_of_hour = df_last_hour.iloc[-1]
                        hour_open = first_candle_of_hour['open']; hour_close = last_candle_of_hour['close']
                        hour_status = "Neutral"; box_border_color = "Yellow"
                        if hour_close > hour_open: hour_status = "Bullish"; box_border_color = "lightgreen"
                        elif hour_close < hour_open: hour_status = "Bearish"; box_border_color = "lightcoral"
                        st.sidebar.info(f"Prev Hr State: {hour_status}")
                        fig.add_shape(type="rect", x0=last_completed_hour_start, x1=last_completed_hour_end, y0=hour_low, y1=hour_high, line=dict(color=box_border_color, width=1), fillcolor="rgba(255, 255, 0, 0.10)", layer="below")
                        st.sidebar.info(f"Time: {last_completed_hour_start.strftime('%H:%M')} - {last_completed_hour_end.strftime('%H:%M')} CT"); st.sidebar.info(f"Range: {hour_low:.2f} - {hour_high:.2f}")
                        first_candle_time = first_candle_of_hour.name; first_candle_high = first_candle_of_hour['high']; first_candle_low = first_candle_of_hour['low']
                        match_info = []
                        if first_candle_high == hour_high: first_candle_high_match_x.append(first_candle_time); first_candle_high_match_y.append(first_candle_high); match_info.append("Imb High")
                        if first_candle_low == hour_low: first_candle_low_match_x.append(first_candle_time); first_candle_low_match_y.append(first_candle_low); match_info.append("Imb Low")
                        if match_info: st.sidebar.info(f"Prev Hr Info: {'; '.join(match_info)}")
                    # else: st.sidebar.warning("No data in previous hour for box.") # Reduce noise
                # else: st.sidebar.warning("Not enough data for hour box.") # Reduce noise
            except Exception as e: st.sidebar.error(f"Error processing hourly box/state: {e}")

            if first_candle_high_match_x: fig.add_trace(go.Scatter(x=first_candle_high_match_x, y=first_candle_high_match_y, mode='markers', marker=dict(symbol='star', color='lightgreen', size=10), name='PrevHr 1st=H', showlegend=True, hovertemplate = 'PrevHr Imb High<extra></extra>'))
            if first_candle_low_match_x: fig.add_trace(go.Scatter(x=first_candle_low_match_x, y=first_candle_low_match_y, mode='markers', marker=dict(symbol='star', color='lightcoral', size=10), name='PrevHr 1st=L', showlegend=True, hovertemplate = 'PrevHr Imb Low<extra></extra>'))

            try: # Check First Candle of Latest Hour
                 if not df.empty:
                    latest_time = df.index[-1]; latest_hour_start = latest_time.floor('h')
                    df_latest_hour = df.loc[df.index >= latest_hour_start]
                    if not df_latest_hour.empty and len(df_latest_hour) > 0:
                        latest_hour_high = df_latest_hour['high'].max(); latest_hour_low = df_latest_hour['low'].min()
                        first_candle_latest = df_latest_hour.iloc[0]
                        first_candle_latest_time = first_candle_latest.name; first_candle_latest_high = first_candle_latest['high']; first_candle_latest_low = first_candle_latest['low']
                        latest_match_info = []
                        if first_candle_latest_high == latest_hour_high: latest_hr_high_match_x.append(first_candle_latest_time); latest_hr_high_match_y.append(first_candle_latest_high); latest_match_info.append("High")
                        if first_candle_latest_low == latest_hour_low: latest_hr_low_match_x.append(first_candle_latest_time); latest_hr_low_match_y.append(first_candle_latest_low); latest_match_info.append("Low")
                        if latest_match_info: st.sidebar.info(f"Last Hr Info: Imb {'; '.join(latest_match_info)}")
            except Exception as e: st.sidebar.error(f"Error checking latest hour first candle: {e}")

            if latest_hr_high_match_x: fig.add_trace(go.Scatter(x=latest_hr_high_match_x, y=latest_hr_high_match_y, mode='markers', marker=dict(symbol='circle', color='lightblue', size=8, line=dict(color='white', width=1)), name='LatestHr 1st=H', showlegend=True, hovertemplate = 'LatestHr 1st=High<extra></extra>'))
            if latest_hr_low_match_x: fig.add_trace(go.Scatter(x=latest_hr_low_match_x, y=latest_hr_low_match_y, mode='markers', marker=dict(symbol='circle', color='purple', size=8, line=dict(color='white', width=1)), name='LatestHr 1st=L', showlegend=True, hovertemplate = 'LatestHr 1st=Low<extra></extra>'))

            try: # Highlight 32m Candle
                if granularity == 'M1' and not df.empty:
                    latest_time_m32 = df.index[-1]; latest_hour_start_m32 = latest_time_m32.floor('h')
                    target_time_m32 = latest_hour_start_m32 + timedelta(minutes=32)
                    if target_time_m32 in df.index:
                        target_candle_data = df.loc[target_time_m32]; marker_y_pos = target_candle_data['high'] * 1.0005
                        fig.add_trace(go.Scatter(x=[target_time_m32], y=[marker_y_pos], mode='markers', marker=dict(symbol='circle', color='red', size=7), name='32m Mark', showlegend=False, hovertemplate = f'32nd min ({target_time_m32.strftime("%H:%M")})<extra></extra>'))
                        # st.sidebar.info(f"Marked 32m candle ({target_time_m32.strftime('%H:%M')} CT).") # Removed for less noise
            except Exception as e: st.sidebar.error(f"Error marking 32m candle: {e}")

            if hour_high is not None and hour_low is not None and last_completed_hour_end is not None: # Breakout Marker Logic
                try:
                    df_after_hour = df.loc[df.index >= last_completed_hour_end]
                    if not df_after_hour.empty:
                        first_upside_breakout_time, first_upside_breakout_high = None, None; first_downside_breakout_time, first_downside_breakout_low = None, None
                        for index, row in df_after_hour.iterrows():
                            if first_upside_breakout_time is None and row['high'] > hour_high: first_upside_breakout_time, first_upside_breakout_high = index, row['high']
                            if first_downside_breakout_time is None and row['low'] < hour_low: first_downside_breakout_time, first_downside_breakout_low = index, row['low']
                            if first_upside_breakout_time and first_downside_breakout_time: break
                        if first_upside_breakout_time: fig.add_trace(go.Scatter(x=[first_upside_breakout_time], y=[first_upside_breakout_high], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Upside Breakout', showlegend=True)); st.sidebar.info(f"Upside BO at {first_upside_breakout_time.strftime('%H:%M')} CT") # Shorter
                        if first_downside_breakout_time: fig.add_trace(go.Scatter(x=[first_downside_breakout_time], y=[first_downside_breakout_low], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Downside Breakout', showlegend=True)); st.sidebar.info(f"Downside BO at {first_downside_breakout_time.strftime('%H:%M')} CT") # Shorter
                        if not first_upside_breakout_time and not first_downside_breakout_time: st.sidebar.info("No breakout yet.")
                    # else: st.sidebar.info("No candles after completed hour.") # Removed for less noise
                except Exception as e: st.sidebar.error(f"Error checking breakouts: {e}")

            try: # Hourly Open Line Segment Logic
                if not df.empty:
                    latest_time_line = df.index[-1]; latest_hour_start = latest_time_line.floor('h')
                    first_candle_series = df.loc[df.index >= latest_hour_start]
                    if not first_candle_series.empty:
                        first_candle = first_candle_series.iloc[0]; latest_hour_open_price = first_candle['open']
                        start_time_of_line = first_candle.name; end_time_of_line = df.index[-1]
                        if start_time_of_line != end_time_of_line:
                            fig.add_shape(type="line", x0=start_time_of_line, y0=latest_hour_open_price, x1=end_time_of_line, y1=latest_hour_open_price, line=dict(color="yellow", width=1, dash="dash"), layer="below")
                            fig.add_annotation(x=end_time_of_line, y=latest_hour_open_price, xref="x", yref="y", text=f"Open: {latest_hour_open_price:.2f}", showarrow=False, yshift=5, xanchor="right", font=dict(size=10, color="yellow"))
                            st.sidebar.info(f"Open at {latest_hour_start.strftime('%H:%M')} CT: {latest_hour_open_price:.2f}") # Added CT back for clarity
            except IndexError: st.sidebar.warning("Not enough data for line.")
            except Exception as e: st.sidebar.error(f"Error drawing line segment: {e}")

            fig.update_layout(title=f'{instrument} ({granularity}) Candlestick Chart', xaxis_title=f'Time ({TARGET_TIMEZONE_STR})', yaxis_title='Price', xaxis_rangeslider_visible=False, template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), width=1000, height=1000, hovermode='x unified', xaxis=dict(title=f'Time ({TARGET_TIMEZONE_STR})', showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot', spikecolor='grey', spikethickness=1), yaxis=dict(title='Price', showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot', spikecolor='grey', spikethickness=1))

            st.plotly_chart(fig)
            # --- END PASTE of Charting Logic ---

        else:
            st.error("Failed to load or process data.")
else:
    st.info("Enter details and click 'Load Chart'.")