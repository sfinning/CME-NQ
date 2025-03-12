import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path):
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
    # Load the CSV file
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows of daily data")
    
    # Convert timestamp to datetime
    if 'ts_event' in df.columns:
        sample_value = df['ts_event'].iloc[0]
        unit = 'ns' if len(str(int(sample_value))) > 13 else 'ms'
        df['date'] = pd.to_datetime(df['ts_event'], unit=unit)
    else:
        print("Error: Could not find timestamp column")
        return None
    
    # Add day of week
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['date'].dt.day_name()
    
    # Filter out rows with missing or zero volume (likely non-trading days)
    df = df[df['volume'] > 0]
    
    # Filter for the main continuous contract (usually has the highest volume)
    # Group by date and select the row with the highest volume for each date
    df = df.sort_values(['date', 'volume'], ascending=[True, False])
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    return df

def create_weekly_data(df):
    print("Creating weekly data...")
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create a week identifier (year + week number)
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
    
    # Group by week and aggregate
    weekly_df = df.groupby('year_week').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'date': 'first'  # Get the first day of the week
    }).reset_index()
    
    # Identify bullish weeks (close > open)
    weekly_df['is_bullish'] = weekly_df['close'] > weekly_df['open']
    
    print(f"Created {len(weekly_df):,} weeks")
    bullish_count = weekly_df['is_bullish'].sum()
    print(f"Found {bullish_count:,} bullish weeks ({bullish_count/len(weekly_df)*100:.1f}%)")
    
    return weekly_df

def analyze_weekly_lows(df, weekly_df):
    print("Analyzing which day of week creates the low in bullish weeks...")
    
    results = []
    
    # For each bullish week
    for _, week in weekly_df[weekly_df['is_bullish']].iterrows():
        week_start = week['date']
        week_end = week_start + timedelta(days=7)
        
        # Get all days in this week
        days_in_week = df[(df['date'] >= week_start) & (df['date'] < week_end)]
        
        if len(days_in_week) == 0:
            continue
        
        # Find the day(s) with the low for the week
        week_low = week['low']
        low_days = days_in_week[days_in_week['low'] == week_low]
        
        if len(low_days) > 0:
            # Take the first day if multiple days have the same low
            low_day = low_days.iloc[0]
            
            results.append({
                'week_start': week_start,
                'low_day_of_week': low_day['day_of_week'],
                'low_day_name': low_day['day_name']
            })
    
    results_df = pd.DataFrame(results)
    print(f"Successfully analyzed {len(results_df):,} bullish weeks")
    
    return results_df

def analyze_and_visualize_results(results_df):
    # Count occurrences of each day
    day_counts = results_df['low_day_name'].value_counts()
    total_weeks = len(results_df)
    
    # Calculate probabilities
    day_probabilities = (day_counts / total_weeks * 100).round(1)
    
    # Sort by day of week (not alphabetically)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_probabilities = day_probabilities.reindex(day_order)
    
    print("\nProbability of each day of week setting the low in bullish weeks:")
    for day, probability in day_probabilities.items():
        count = day_counts[day]
        print(f"{day}: {count:,} occurrences ({probability:.1f}%)")
    
    # Identify the day with highest probability
    highest_prob_day = day_probabilities.idxmax()
    highest_prob = day_probabilities.max()
    
    print(f"\nThe day of the week with the highest probability of setting the low in a bullish week is:")
    print(f"{highest_prob_day} with {highest_prob:.1f}% probability")
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(day_probabilities.index, day_probabilities.values, color='skyblue')
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 1, 
            f'{height:.1f}%', 
            ha='center', 
            va='bottom'
        )
    
    plt.title('Probability of Each Day Setting the Low in Bullish Weeks', fontsize=14)
    plt.ylabel('Probability (%)', fontsize=12)
    plt.ylim(0, max(day_probabilities.values) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    try:
        plt.savefig('weekly_low_probability_by_day.png')
        print("Chart saved as 'weekly_low_probability_by_day.png'")
    except Exception as e:
        print(f"Could not save chart: {e}")
    
    return highest_prob_day, highest_prob

def main():
    # Try different potential file paths
    file_paths = [
        "nq-ohlcv-1d.csv",
        r"c:\sqlite\CME-NQ\nq-ohlcv-1d.csv",
        os.path.join(os.path.dirname(__file__), "nq-ohlcv-1d.csv")
    ]
    
    # Try to find the file
    df = None
    for path in file_paths:
        if os.path.exists(path):
            print(f"Found file at: {path}")
            df = load_and_prepare_data(path)
            if df is not None:
                break
    
    # If still not found, ask for manual path
    if df is None:
        print("Could not find the data file. Please enter the full path:")
        user_path = input("> ")
        if os.path.exists(user_path):
            df = load_and_prepare_data(user_path)
        else:
            print("File not found. Exiting.")
            return
    
    if df is None:
        print("Could not load data. Exiting.")
        return
    
    # Create weekly data
    weekly_df = create_weekly_data(df)
    
    # Analyze which day of week sets the low in bullish weeks
    results_df = analyze_weekly_lows(df, weekly_df)
    
    # Analyze and visualize results
    analyze_and_visualize_results(results_df)

if __name__ == "__main__":
    main()