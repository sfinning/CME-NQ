import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time # Required if using auto-refresh loop later
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError # Import zoneinfo

# --- Configuration ---
OANDA_API_BASE_URL = "https://api-fxpractice.oanda.com/v3"
DEFAULT_INSTRUMENT = "NAS100_USD"
DEFAULT_GRANULARITY = "M1"
DEFAULT_COUNT = 180
TARGET_TIMEZONE_STR = "America/Chicago" # Define target timezone string

# --- Helper Function to Fetch OANDA Data ---
def fetch_oanda_data(api_key, account_id, instrument, granularity, count):
    """
    Fetches historical candle data from the OANDA v20 API and converts
    the time index to the target timezone using zoneinfo.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {"granularity": granularity, "count": count, "price": "M"}
    url = f"{OANDA_API_BASE_URL}/instruments/{instrument}/candles"
    response_obj = None

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
            time_utc = pd.to_datetime(candle['time']) # Parsed as UTC
            # ... (extract other fields: volume, open, high, low, close) ...
            volume = candle['volume']
            open_price = float(candle['mid']['o'])
            high_price = float(candle['mid']['h'])
            low_price = float(candle['mid']['l'])
            close_price = float(candle['mid']['c'])
            records.append({'time': time_utc, 'open': open_price, 'high': high_price,
                            'low': low_price, 'close': close_price, 'volume': volume})


        if not records:
             st.warning("All received candles were incomplete.")
             return None

        df = pd.DataFrame(records)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True) # df index is UTC here

        # Convert timezone using zoneinfo
        try:
            # Create ZoneInfo object from the target string
            target_tz_obj = ZoneInfo(TARGET_TIMEZONE_STR)
            # Convert the DataFrame index using the ZoneInfo object
            df = df.tz_convert(target_tz_obj)
            st.sidebar.write(f"Timezone set to: {df.index.tz}") # Log actual timezone object used
        except ZoneInfoNotFoundError:
            st.error(f"Timezone '{TARGET_TIMEZONE_STR}' not found by zoneinfo.")
            st.error("Make sure the timezone name is correct and tzdata might be needed (`pip install tzdata`).")
            st.error("Proceeding with UTC times.")
            # Optionally return UTC df or None
            # return None
        except Exception as tz_error:
            st.error(f"Failed to convert timezone using zoneinfo: {tz_error}")
            st.error("Proceeding with UTC times.")
            # Optionally return UTC df or None
            # return None

        return df # Return df with index converted to target timezone (or UTC on error)

    # ... (Rest of error handling remains the same) ...
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
# Use the string variable for display consistency
st.write(f"Connect to OANDA API. Times displayed in {TARGET_TIMEZONE_STR}.")

# --- Sidebar for Inputs ---
# ... (Sidebar remains the same: Credentials, Parameters) ...
st.sidebar.header("OANDA Credentials")
api_key = st.sidebar.text_input("OANDA API Key", value="", type="password", help="Enter your OANDA API access token.")
account_id = st.sidebar.text_input("OANDA Account ID", value="", help="Enter your OANDA Account ID.")
st.sidebar.markdown("---")
st.sidebar.header("Chart Parameters")
instrument = st.sidebar.text_input("Instrument", value=DEFAULT_INSTRUMENT, help="e.g., NAS100_USD, EUR_USD, etc.")
granularity = st.sidebar.selectbox(
    "Timeframe",
    options=['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'h', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'],
    index=4,
    help="Time interval for candles (S=Seconds, M=Minutes, h=Hours, D=Day, W=Week, M=Month)"
)
count = st.sidebar.number_input(
    "Candles", min_value=10, max_value=5000, value=DEFAULT_COUNT, step=10,
    help="How many past candles to fetch (max 5000)."
)

# --- Main Logic ---
if st.sidebar.button("Load Chart"):
    # ... (Input validation remains the same) ...
    if not api_key or not account_id or not instrument:
        if not api_key: st.warning("Please enter your OANDA API Key.")
        if not account_id: st.warning("Please enter your OANDA Account ID.")
        if not instrument: st.warning("Please enter an Instrument.")
    else:
        st.info(f"Fetching {count} candles for {instrument} ({granularity})...")
        # Fetch data (timezone conversion using zoneinfo happens inside)
        df = fetch_oanda_data(api_key, account_id, instrument, granularity, count)

        if df is not None and not df.empty:
            st.success("Data loaded successfully!")
            # The DataFrame 'df' index is now already in the target timezone (or UTC if conversion failed)
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=instrument)])

            # --- Hourly Box, Breakout Marker, Line Segment Logic ---
            # This logic remains exactly the same as before, as it now operates
            # on the DataFrame `df` which already has the correctly timezoned index.
            # All derived timestamps (last_completed_hour_start, breakout times, etc.)
            # will also be in the target timezone.

            # --- Hourly Box Logic ---
            hour_low, hour_high, last_completed_hour_end = None, None, None
            try:
                if len(df.index) > 1:
                    latest_time_box = df.index[-1]
                    current_hour_start_box = latest_time_box.floor('h')
                    last_completed_hour_start = current_hour_start_box - pd.Timedelta(hours=1)
                    last_completed_hour_end = current_hour_start_box
                    df_last_hour = df.loc[(df.index >= last_completed_hour_start) & (df.index < last_completed_hour_end)]
                    if not df_last_hour.empty:
                        hour_low = df_last_hour['low'].min()
                        hour_high = df_last_hour['high'].max()
                        fig.add_shape(type="rect", x0=last_completed_hour_start, x1=last_completed_hour_end, y0=hour_low, y1=hour_high,
                                      line=dict(color="Yellow", width=1), fillcolor="rgba(255, 255, 0, 0.15)", layer="below")
                        st.sidebar.info(f"Highlighted: {last_completed_hour_start.strftime('%H:%M')} - {last_completed_hour_end.strftime('%H:%M')} CT (R: {hour_low:.2f}-{hour_high:.2f})")
                    else: st.sidebar.warning("No data in previous hour for box.")
                else: st.sidebar.warning("Not enough data for hour box.")
            except Exception as e: st.sidebar.error(f"Error drawing highlight box: {e}")

            # --- Breakout Marker Logic ---
            if hour_high is not None and hour_low is not None and last_completed_hour_end is not None:
                try:
                    df_after_hour = df.loc[df.index >= last_completed_hour_end]
                    if not df_after_hour.empty:
                        # ... (breakout finding logic remains the same) ...
                        first_upside_breakout_time, first_upside_breakout_high = None, None
                        first_downside_breakout_time, first_downside_breakout_low = None, None
                        for index, row in df_after_hour.iterrows():
                            if first_upside_breakout_time is None and row['high'] > hour_high:
                                first_upside_breakout_time, first_upside_breakout_high = index, row['high']
                            if first_downside_breakout_time is None and row['low'] < hour_low:
                                first_downside_breakout_time, first_downside_breakout_low = index, row['low']
                            if first_upside_breakout_time and first_downside_breakout_time: break

                        if first_upside_breakout_time:
                            fig.add_trace(go.Scatter(x=[first_upside_breakout_time], y=[first_upside_breakout_high], mode='markers',
                                                     marker=dict(symbol='triangle-up', color='lime', size=10), name='Upside Breakout', showlegend=True))
                            st.sidebar.info(f"Upside breakout at {first_upside_breakout_time.strftime('%H:%M')} CT")
                        if first_downside_breakout_time:
                            fig.add_trace(go.Scatter(x=[first_downside_breakout_time], y=[first_downside_breakout_low], mode='markers',
                                                     marker=dict(symbol='triangle-down', color='red', size=10), name='Downside Breakout', showlegend=True))
                            st.sidebar.info(f"Downside breakout at {first_downside_breakout_time.strftime('%H:%M')} CT")
                        if not first_upside_breakout_time and not first_downside_breakout_time: st.sidebar.info("No breakout yet.")
                    else: st.sidebar.info("No candles after completed hour.")
                except Exception as e: st.sidebar.error(f"Error checking breakouts: {e}")

            # --- Hourly Open Line Segment Logic ---
            try:
                if not df.empty:
                    latest_time_line = df.index[-1]
                    latest_hour_start = latest_time_line.floor('h')
                    first_candle_series = df.loc[df.index >= latest_hour_start]
                    if not first_candle_series.empty:
                        # ... (line segment logic remains the same) ...
                        first_candle = first_candle_series.iloc[0]
                        latest_hour_open_price = first_candle['open']
                        start_time_of_line = first_candle.name
                        end_time_of_line = df.index[-1]
                        if start_time_of_line != end_time_of_line:
                            fig.add_shape(type="line", x0=start_time_of_line, y0=latest_hour_open_price, x1=end_time_of_line, y1=latest_hour_open_price,
                                          line=dict(color="cyan", width=1, dash="dash"), layer="below")
                            fig.add_annotation(x=end_time_of_line, y=latest_hour_open_price, xref="x", yref="y", text=f"Open: {latest_hour_open_price:.2f}",
                                               showarrow=False, yshift=5, xanchor="right", font=dict(size=10, color="cyan"))
                            st.sidebar.info(f"Line at open ({latest_hour_start.strftime('%H:%M')} CT): {latest_hour_open_price:.2f}")
                        else: st.sidebar.info("Line not drawn (hour start = data end).")
                    else: st.sidebar.warning("No first candle found for latest hour.")
                else: st.sidebar.warning("DataFrame empty, cannot draw line.")
            except IndexError: st.sidebar.warning("Not enough data for line.")
            except Exception as e: st.sidebar.error(f"Error drawing line segment: {e}")


            # --- Final Layout Update ---
            fig.update_layout(
                title=f'{instrument} ({granularity}) Candlestick Chart',
                xaxis_title=f'Time ({TARGET_TIMEZONE_STR})', # Use string for label
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                width=1000,
                height=1000
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig) # Fixed dimensions, no use_container_width

            # Optional Raw Data Display (currently commented out)
            # st.subheader("Raw Data")
            # st.dataframe(df.tail())

        else:
            st.error("Failed to load or process data. Check inputs and API response details above.")
else:
    st.info("Enter your OANDA credentials and parameters in the sidebar, then click 'Load Chart'.")