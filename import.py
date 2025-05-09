import pandas as pd
from datetime import time, datetime as dt_datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# --- Configuration ---
CSV_URL = "https://media.githubusercontent.com/media/sfinning/CME-NQ/refs/heads/main/nq-ohlcv-1m.csv"
TARGET_TIMEZONE_STR = "America/Chicago" # Assuming CST/CDT for 05:04, 06:04, 07:04
OPEN_PRICE_TIME = time(5, 4)
CHECK_WINDOW_START_TIME = time(6, 4)
CHECK_WINDOW_END_TIME = time(7, 4)

def analyze_price_return():
    """
    Analyzes NQ futures data to find days where the price returns to the 05:04 open
    price between 06:04 and 07:04 in the specified timezone.
    """
    try:
        # --- 1. Load Data ---
        print(f"Loading data from {CSV_URL}...")
        df = pd.read_csv(CSV_URL)
        print("Data loaded successfully.")

        # --- 2. Timezone and Timestamp Conversion ---
        try:
            target_zone = ZoneInfo(TARGET_TIMEZONE_STR)
        except ZoneInfoNotFoundError:
            print(f"ERROR: Timezone '{TARGET_TIMEZONE_STR}' not found. "
                  "Please ensure your system's timezone database is up to date or use a valid IANA timezone name.")
            return

        # Convert ts_event from nanoseconds to UTC datetime objects
        df['ts_event_utc'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)

        # Convert UTC timestamps to the target local timezone
        df['local_time'] = df['ts_event_utc'].dt.tz_convert(target_zone)

        # Extract date and time for easier filtering
        df['local_date'] = df['local_time'].dt.date
        df['local_time_of_day'] = df['local_time'].dt.time

        print(f"Timestamps converted. Target timezone: {TARGET_TIMEZONE_STR}")
        # print(df[['ts_event', 'ts_event_utc', 'local_time', 'local_date', 'local_time_of_day']].head())


        # --- 3. Analysis ---
        results = []
        unique_dates = sorted(df['local_date'].unique())
        print(f"Analyzing {len(unique_dates)} unique dates...")

        for current_date in unique_dates:
            daily_data = df[df['local_date'] == current_date]

            # --- Find 05:04 Open Price ---
            # Filter for records at exactly OPEN_PRICE_TIME
            open_price_minute_data = daily_data[daily_data['local_time_of_day'] == OPEN_PRICE_TIME]

            if open_price_minute_data.empty:
                # print(f"No data at {OPEN_PRICE_TIME.strftime('%H:%M')} for {current_date}")
                continue

            # If multiple symbols, pick one with highest volume
            if len(open_price_minute_data) > 1:
                target_row_0504 = open_price_minute_data.loc[open_price_minute_data['volume'].idxmax()]
            else:
                target_row_0504 = open_price_minute_data.iloc[0]

            target_open_price = target_row_0504['open']
            target_symbol = target_row_0504['symbol']
            # print(f"Date: {current_date}, Symbol: {target_symbol}, 05:04 Open: {target_open_price}")


            # --- Check Window (06:04 to 07:04 local time) ---
            # Define the window in local time for the current date
            # Note: dt_datetime.combine needs a date object, not a pandas Timestamp date
            py_current_date = pd.Timestamp(current_date).to_pydatetime().date() # Convert pandas date to python date

            start_check_dt_local = dt_datetime.combine(py_current_date, CHECK_WINDOW_START_TIME, tzinfo=target_zone)
            end_check_dt_local = dt_datetime.combine(py_current_date, CHECK_WINDOW_END_TIME, tzinfo=target_zone)

            # Convert window boundaries to UTC for filtering ts_event_utc
            start_check_dt_utc = start_check_dt_local.astimezone(timezone.utc)
            end_check_dt_utc = end_check_dt_local.astimezone(timezone.utc)
            
            # Filter for the specific symbol and the UTC time window
            window_data = daily_data[
                (daily_data['symbol'] == target_symbol) &
                (daily_data['ts_event_utc'] >= start_check_dt_utc) &
                (daily_data['ts_event_utc'] <= end_check_dt_utc)
            ]

            if window_data.empty:
                # print(f"No data for symbol {target_symbol} in window {CHECK_WINDOW_START_TIME}-{CHECK_WINDOW_END_TIME} on {current_date}")
                continue

            # Check if the price returned to the target_open_price
            # This means the target_open_price is between the low and high of any bar in the window
            returned_to_price = window_data[
                (window_data['low'] <= target_open_price) &
                (window_data['high'] >= target_open_price)
            ]

            if not returned_to_price.empty:
                first_return_time_local = returned_to_price.iloc[0]['local_time']
                results.append({
                    "date": current_date.strftime('%Y-%m-%d'),
                    "target_symbol": target_symbol,
                    "open_price_0504": target_open_price,
                    "returned_at_local": first_return_time_local.strftime('%H:%M:%S %Z%z'),
                    "returned_low": returned_to_price.iloc[0]['low'],
                    "returned_high": returned_to_price.iloc[0]['high'],
                })

        # --- 4. Output Results ---
        if results:
            print("\n--- Analysis Results ---")
            print(f"Found {len(results)} instances where the price returned to the 05:04 open between "
                  f"{CHECK_WINDOW_START_TIME.strftime('%H:%M')} and {CHECK_WINDOW_END_TIME.strftime('%H:%M')} ({TARGET_TIMEZONE_STR}):")
            results_df = pd.DataFrame(results)
            print(results_df.to_string())
        else:
            print("\n--- Analysis Results ---")
            print("No instances found where the price returned to the 05:04 open under the specified conditions.")

    except FileNotFoundError:
        print(f"ERROR: The file {CSV_URL} was not found.")
    except pd.errors.EmptyDataError:
        print(f"ERROR: The file {CSV_URL} is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    analyze_price_return()
