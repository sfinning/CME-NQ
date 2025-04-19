import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import datetime, date, time as dt_time, timedelta

# --- Configuration ---
OANDA_API_BASE_URL = "https://api-fxpractice.oanda.com/v3"
DEFAULT_INSTRUMENT = "NAS100_USD"
DEFAULT_GRANULARITY = "M1"
DEFAULT_COUNT = 180
TARGET_TIMEZONE_STR = "America/Chicago"

# --- Helper Function to Fetch OANDA Data ---
# (Remains the same as the previous version - includes start_time logic and tz conversion)
def fetch_oanda_data(api_key, account_id, instrument, granularity, count, start_time_utc_str=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{OANDA_API_BASE_URL}/instruments/{instrument}/candles"
    response_obj = None
    params = {"granularity": granularity, "price": "M"}
    if start_time_utc_str:
        params["from"] = start_time_utc_str; params["count"] = count
        st.sidebar.caption(f"Fetching {count} candles from {start_time_utc_str[:10]} ...")
    else:
        params["count"] = count
        st.sidebar.caption(f"Fetching latest {count} candles")
    try:
        response_obj = requests.get(url, headers=headers, params=params)
        response_obj.raise_for_status(); data = response_obj.json()
        if 'candles' not in data or not data['candles']: st.warning(f"No candle data for {instrument} ({granularity})."); return None
        records = []
        for candle in data['candles']:
            if not candle['complete']: continue
            time_utc = pd.to_datetime(candle['time'])
            volume = candle['volume']; open_price = float(candle['mid']['o']); high_price = float(candle['mid']['h'])
            low_price = float(candle['mid']['l']); close_price = float(candle['mid']['c'])
            records.append({'time': time_utc, 'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price, 'volume': volume})
        if not records: st.warning("All candles incomplete."); return None
        df = pd.DataFrame(records); df.set_index('time', inplace=True); df.sort_index(inplace=True)
        try:
            target_tz_obj = ZoneInfo(TARGET_TIMEZONE_STR); df = df.tz_convert(target_tz_obj)
        except Exception as tz_error: st.error(f"TZ Conversion Error: {tz_error}. Using UTC.");
        return df
    except requests.exceptions.RequestException as e: st.error(f"API Error: {e}");
    except Exception as e: st.error(f"Data Fetch Error: {e}");
    return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("📈 Breakout & RTO Analysis")

# --- Sidebar for Inputs ---
st.sidebar.header("OANDA Credentials")
api_key = st.sidebar.text_input("API Key", value="", type="password")
account_id = st.sidebar.text_input("Account ID", value="")
st.sidebar.markdown("---")
st.sidebar.header("Chart Parameters")
instrument = st.sidebar.text_input("Instrument", value=DEFAULT_INSTRUMENT)
granularity = st.sidebar.selectbox("Timeframe", ['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'h', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'], index=4)
count = st.sidebar.number_input("Candles", min_value=10, max_value=5000, value=DEFAULT_COUNT, step=10)
st.sidebar.markdown("---")
use_custom_start = st.sidebar.checkbox("Custom Start Time", value=False)
start_time_api_str = None
if use_custom_start:
    try: target_tz = ZoneInfo(TARGET_TIMEZONE_STR); now_local = datetime.now(target_tz)
    except Exception: now_local = datetime.now()
    start_date = st.sidebar.date_input("Start Date", value=now_local.date())
    start_time = st.sidebar.time_input(
        "Start Time (CT)",
        value=dt_time(0, 0),
        step=3600
    )
    if start_date and start_time:
        try:
            start_dt_naive = datetime.combine(start_date, start_time)
            start_dt_aware_target = start_dt_naive.replace(tzinfo=target_tz)
            start_dt_utc = start_dt_aware_target.astimezone(ZoneInfo("UTC"))
            start_time_api_str = start_dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        except Exception as e: st.sidebar.error(f"Start time processing error: {e}"); start_time_api_str = None

# --- Main Logic ---
if st.sidebar.button("Load Chart"):
    if not api_key or not account_id or not instrument:
        if not api_key: st.warning("API Key needed.")
        if not account_id: st.warning("Account ID needed.")
        if not instrument: st.warning("Instrument needed.")
    else:
        st.info(f"Requesting data for {instrument} ({granularity}) from OANDA API...")
        df = fetch_oanda_data(api_key, account_id, instrument, granularity, count, start_time_api_str)

        if df is not None and not df.empty:
            # --- Display Latest Date Info ---
            try:
                latest_timestamp = df.index[-1]; day_of_week = latest_timestamp.strftime('%A'); date_str = latest_timestamp.strftime('%B %d, %Y')
                st.sidebar.markdown("---");
                st.sidebar.info(f"{day_of_week} {date_str}");
            except Exception: pass

            # --- Create Plotly Chart ---
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=instrument, increasing=dict(line=dict(color='royalblue'), fillcolor='royalblue'), decreasing=dict(line=dict(color='lightgrey'), fillcolor='lightgrey'))])

            # Initialize variables
            hour_low, hour_high, last_completed_hour_end = None, None, None
            first_candle_high_match_x, first_candle_high_match_y = [], []; first_candle_low_match_x, first_candle_low_match_y = [], []
            latest_hr_high_match_x, latest_hr_high_match_y = [], []; latest_hr_low_match_x, latest_hr_low_match_y = [], []
            latest_hour_open_price = None; first_candle_latest_time = None # Initialize here
            return_marker_x, return_marker_y = [], []

            # --- Calculate Latest Hour Open Price & First Candle Time ---
            try:
                if not df.empty:
                    latest_time_line = df.index[-1]; latest_hour_start = latest_time_line.floor('h')
                    first_candle_series = df.loc[df.index >= latest_hour_start]
                    if not first_candle_series.empty:
                        latest_hour_open_price = first_candle_series.iloc[0]['open']
                        first_candle_latest_time = first_candle_series.iloc[0].name # Get timestamp
            except Exception as e: st.sidebar.error(f"Could not calculate latest hour info: {e}")


            # --- Hourly Box Logic (Last Completed Hour) & First Candle Check ---
            try:
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
                        st.sidebar.info(f"Time: {last_completed_hour_start.strftime('%H:%M')}-{last_completed_hour_end.strftime('%H:%M')} CT")
                        st.sidebar.info(f"Range: {hour_low:.2f}-{hour_high:.2f}")
                        first_candle_time = first_candle_of_hour.name; first_candle_high = first_candle_of_hour['high']; first_candle_low = first_candle_of_hour['low']
                        match_info = []
                        if first_candle_high == hour_high: first_candle_high_match_x.append(first_candle_time); first_candle_high_match_y.append(first_candle_high); match_info.append("Imb High")
                        if first_candle_low == hour_low: first_candle_low_match_x.append(first_candle_time); first_candle_low_match_y.append(first_candle_low); match_info.append("Imb Low")
                        if match_info: st.sidebar.info(f"Prev Hr Info: {'; '.join(match_info)}")
            except Exception as e: st.sidebar.error(f"Error processing hourly box/state: {e}")

            # --- Markers for First Candle H/L Match (Completed Hour) ---
            if first_candle_high_match_x: fig.add_trace(go.Scatter(x=first_candle_high_match_x, y=first_candle_high_match_y, mode='markers', marker=dict(symbol='star', color='lightgreen', size=10), name='Prev Hr Imb High', showlegend=True, hovertemplate = 'Prev Hr Imb High<extra></extra>'))
            if first_candle_low_match_x: fig.add_trace(go.Scatter(x=first_candle_low_match_x, y=first_candle_low_match_y, mode='markers', marker=dict(symbol='star', color='lightcoral', size=10), name='Prev Hr Imb Low', showlegend=True, hovertemplate = 'Prev Hr Imb Low<extra></extra>'))

            # --- Check First Candle of Latest Hour vs Its Range So Far ---
            try:
                 if not df.empty and first_candle_latest_time is not None: # Need first candle time here
                    latest_hour_start = first_candle_latest_time.floor('h') # Recalculate based on actual first candle time
                    df_latest_hour = df.loc[df.index >= latest_hour_start] # Filter from actual start
                    if not df_latest_hour.empty and len(df_latest_hour) > 0:
                        latest_hour_high = df_latest_hour['high'].max(); latest_hour_low = df_latest_hour['low'].min()
                        first_candle_latest = df_latest_hour.loc[first_candle_latest_time] # Access by known index
                        first_candle_latest_high = first_candle_latest['high']; first_candle_latest_low = first_candle_latest['low']
                        latest_match_info = []
                        if first_candle_latest_high == latest_hour_high: latest_hr_high_match_x.append(first_candle_latest_time); latest_hr_high_match_y.append(first_candle_latest_high); latest_match_info.append("High")
                        if first_candle_latest_low == latest_hour_low: latest_hr_low_match_x.append(first_candle_latest_time); latest_hr_low_match_y.append(first_candle_latest_low); latest_match_info.append("Low")
                        if latest_match_info: st.sidebar.info(f"Last Hr Info: Imb {'; '.join(latest_match_info)}")
            except Exception as e: st.sidebar.error(f"Error checking latest hour 1st candle: {e}")

            # --- Markers for First Candle H/L Match (Latest Hour) ---
            if latest_hr_high_match_x: fig.add_trace(go.Scatter(x=latest_hr_high_match_x, y=latest_hr_high_match_y, mode='markers', marker=dict(symbol='circle', color='lightblue', size=8, line=dict(color='white', width=1)), name='Last Hr Imb High', showlegend=True, hovertemplate = 'Last Hr Imb High<extra></extra>'))
            if latest_hr_low_match_x: fig.add_trace(go.Scatter(x=latest_hr_low_match_x, y=latest_hr_low_match_y, mode='markers', marker=dict(symbol='circle', color='purple', size=8, line=dict(color='white', width=1)), name='Last Hr Imb Low', showlegend=True, hovertemplate = 'Last Hr Imb Low<extra></extra>'))

            # --- Highlight 32nd Minute Candle ---
            # (Remains the same)
            try:
                if granularity == 'M1' and not df.empty:
                    latest_time_m32 = df.index[-1]; latest_hour_start_m32 = latest_time_m32.floor('h'); target_time_m32 = latest_hour_start_m32 + timedelta(minutes=32)
                    if target_time_m32 in df.index:
                        target_candle_data = df.loc[target_time_m32]; marker_y_pos = target_candle_data['high'] * 1.0005
                        fig.add_trace(go.Scatter(x=[target_time_m32], y=[marker_y_pos], mode='markers', marker=dict(symbol='circle', color='red', size=7), name='32m Mark', showlegend=False, hovertemplate = f'32nd min ({target_time_m32.strftime("%H:%M")})<extra></extra>'))
            except Exception as e: st.sidebar.error(f"Error marking 32m candle: {e}")


            # --- Breakout & Return-to-Open Logic (MODIFIED)---
            if hour_high is not None and hour_low is not None and last_completed_hour_end is not None and latest_hour_open_price is not None:
                try:
                    df_after_hour = df.loc[df.index >= last_completed_hour_end]
                    if not df_after_hour.empty:
                        breakout_occurred = False; return_to_open_marked = False
                        first_upside_breakout_time = None; first_downside_breakout_time = None

                        for index, row in df_after_hour.iterrows():
                            # Check for first breakouts
                            if first_upside_breakout_time is None and row['high'] > hour_high:
                                first_upside_breakout_time = index; breakout_occurred = True
                            if first_downside_breakout_time is None and row['low'] < hour_low:
                                first_downside_breakout_time = index; breakout_occurred = True

                            # Check for return to open *after* a breakout occurred,
                            # *and* *excluding* the first candle of the latest hour
                            if breakout_occurred and not return_to_open_marked:
                                # --- Add condition here ---
                                if first_candle_latest_time is not None and index != first_candle_latest_time:
                                    # --- End Add condition ---
                                    if row['low'] <= latest_hour_open_price <= row['high']:
                                        return_marker_x.append(index)
                                        return_marker_y.append(latest_hour_open_price)
                                        return_to_open_marked = True
                                        # Sidebar message moved after loop

                        # --- Add Markers AFTER the loop ---
                        if first_upside_breakout_time:
                            upside_bo_high = df.loc[first_upside_breakout_time, 'high']
                            fig.add_trace(go.Scatter(x=[first_upside_breakout_time], y=[upside_bo_high], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Bullish Breakout', showlegend=True))
                            st.sidebar.info(f"Bullish BO at {first_upside_breakout_time.strftime('%H:%M')} CT")
                        if first_downside_breakout_time:
                            downside_bo_low = df.loc[first_downside_breakout_time, 'low']
                            fig.add_trace(go.Scatter(x=[first_downside_breakout_time], y=[downside_bo_low], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Bearish Breakout', showlegend=True))
                            st.sidebar.info(f"Bearish BO at {first_downside_breakout_time.strftime('%H:%M')} CT")

                        if return_marker_x: # Check if list has data
                            fig.add_trace(go.Scatter(x=return_marker_x, y=return_marker_y, mode='markers', marker=dict(symbol='diamond', color='magenta', size=10, line=dict(color='white', width=1)), name='RTO', showlegend=True, hovertemplate='RTO<extra></extra>'))
                            st.sidebar.info(f"RTO at {return_marker_x[0].strftime('%H:%M')} CT") # Show time of first return

                        if not breakout_occurred: st.sidebar.info("No breakout yet.")

                except Exception as e: st.sidebar.error(f"Error checking breakouts/return: {e}")
            elif latest_hour_open_price is None: st.sidebar.warning("Cannot check return: Latest hour open missing.")


            # --- Hourly Open Line Segment Logic ---
            # (Moved calculation earlier, just add shape/annotation here)
            try:
                if latest_hour_open_price is not None and first_candle_latest_time is not None and not df.empty:
                     start_time_of_line = first_candle_latest_time; end_time_of_line = df.index[-1]
                     if start_time_of_line != end_time_of_line:
                         fig.add_shape(type="line", x0=start_time_of_line, y0=latest_hour_open_price, x1=end_time_of_line, y1=latest_hour_open_price, line=dict(color="yellow", width=1, dash="dash"), layer="below")
                         fig.add_annotation(x=end_time_of_line, y=latest_hour_open_price, xref="x", yref="y", text=f"Open: {latest_hour_open_price:.2f}", showarrow=False, yshift=5, xanchor="right", font=dict(size=10, color="yellow"))
                         st.sidebar.info(f"Open at {start_time_of_line.strftime('%H:%M')} CT: {latest_hour_open_price:.2f}")
            except Exception as e: st.sidebar.error(f"Error drawing line segment: {e}")

            # --- Final Layout Update ---
            fig.update_layout(title=f'{instrument} ({granularity}) Chart', xaxis_title=f'Time ({TARGET_TIMEZONE_STR})', yaxis_title='Price', xaxis_rangeslider_visible=False, template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), width=1000, height=1000, hovermode='x unified', xaxis=dict(title=f'Time ({TARGET_TIMEZONE_STR})', showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot', spikecolor='grey', spikethickness=1), yaxis=dict(title='Price', showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot', spikecolor='grey', spikethickness=1))

            # --- Display Chart ---
            st.plotly_chart(fig)

        else:
            st.error("Failed to load or process data.")
else:
    st.info("Enter details and click 'Load Chart'.")