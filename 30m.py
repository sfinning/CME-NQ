import pandas as pd

# Load data
df = pd.read_csv('c:/sqlite/CME-NQ/USATECH.IDXUSD_Candlestick_30_M_BID_30.05.2022-30.05.2025.csv', parse_dates=['Gmt time'])

# Set datetime as index for easier slicing
df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index('Gmt time')

# Function to get daily session high/low
def get_session_high_low(df):
    results = []
    # Get unique dates
    for date in df.index.normalize().unique():
        # Session: 20:30 GMT (current day) to 08:30 GMT (next day)
        start = date + pd.Timedelta(hours=20, minutes=30)
        end = (date + pd.Timedelta(days=1)) + pd.Timedelta(hours=8, minutes=30)
        session = df[(df.index >= start) & (df.index < end)]
        if not session.empty:
            results.append({
                'Session Date': date.strftime('%Y-%m-%d'),
                'Session High': session['High'].max(),
                'Session Low': session['Low'].min()
            })
    return pd.DataFrame(results)

session_ranges = get_session_high_low(df)
print(session_ranges)

def probability_high_low_traded_through(df):
    total = 0
    high_traded_count = 0
    low_traded_count = 0
    both_traded_count = 0
    for date in df.index.normalize().unique():
        # Session 1: 20:30 GMT (current day) to 08:30 GMT (next day)
        session1_start = date + pd.Timedelta(hours=20, minutes=30)
        session1_end = (date + pd.Timedelta(days=1)) + pd.Timedelta(hours=8, minutes=30)
        session1 = df[(df.index >= session1_start) & (df.index < session1_end)]
        if session1.empty:
            continue
        session_high = session1['High'].max()
        session_low = session1['Low'].min()

        # Session 2: 08:30 GMT to 16:30 GMT (next day)
        session2_start = session1_end
        session2_end = (date + pd.Timedelta(days=1)) + pd.Timedelta(hours=16, minutes=30)
        session2 = df[(df.index >= session2_start) & (df.index < session2_end)]
        if session2.empty:
            continue

        high_traded = (session2['High'] > session_high).any()
        low_traded = (session2['Low'] < session_low).any()

        total += 1
        if high_traded:
            high_traded_count += 1
        if low_traded:
            low_traded_count += 1
        if high_traded and low_traded:
            both_traded_count += 1

    print(f"Probability high is traded through: {high_traded_count/total:.2%} ({high_traded_count}/{total})")
    print(f"Probability low is traded through: {low_traded_count/total:.2%} ({low_traded_count}/{total})")
    print(f"Probability both high and low are traded through: {both_traded_count/total:.2%} ({both_traded_count}/{total})")

probability_high_low_traded_through(df)

def probability_high_low_traded_through_by_weekday(df):
    stats = {i: {'total': 0, 'high': 0, 'low': 0, 'both': 0} for i in range(7)}
    for date in df.index.normalize().unique():
        weekday = date.weekday()
        # Session 1: 20:30 GMT (current day) to 08:30 GMT (next day)
        session1_start = date + pd.Timedelta(hours=20, minutes=30)
        session1_end = (date + pd.Timedelta(days=1)) + pd.Timedelta(hours=8, minutes=30)
        session1 = df[(df.index >= session1_start) & (df.index < session1_end)]
        if session1.empty:
            continue
        session_high = session1['High'].max()
        session_low = session1['Low'].min()

        # For Friday, skip to Sunday for the next session
        if weekday == 4:
            # Find the next Sunday in the data
            next_sunday = date + pd.Timedelta(days=(6 - weekday + 1))  # Sunday after Friday
            session2_start = next_sunday + pd.Timedelta(hours=8, minutes=30)
            session2_end = next_sunday + pd.Timedelta(hours=16, minutes=30)
        else:
            # Session 2: 08:30 GMT to 16:30 GMT (next day)
            session2_start = session1_end
            session2_end = (date + pd.Timedelta(days=1)) + pd.Timedelta(hours=16, minutes=30)

        session2 = df[(df.index >= session2_start) & (df.index < session2_end)]
        if session2.empty:
            continue

        high_traded = (session2['High'] > session_high).any()
        low_traded = (session2['Low'] < session_low).any()

        stats[weekday]['total'] += 1
        if high_traded:
            stats[weekday]['high'] += 1
        if low_traded:
            stats[weekday]['low'] += 1
        if high_traded and low_traded:
            stats[weekday]['both'] += 1

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i in range(5):  # Only show Monday to Friday
        total = stats[i]['total']
        if total == 0:
            continue
        print(f"{days[i]}:")
        print(f"  Probability high is traded through: {stats[i]['high']/total:.2%} ({stats[i]['high']}/{total})")
        print(f"  Probability low is traded through: {stats[i]['low']/total:.2%} ({stats[i]['low']}/{total})")
        print(f"  Probability both high and low are traded through: {stats[i]['both']/total:.2%} ({stats[i]['both']}/{total})")

probability_high_low_traded_through_by_weekday(df)