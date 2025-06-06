import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import pytz

def load_and_prepare_data(file_path):
    """Load CSV file and prepare the data for analysis."""
    # Read CSV data
    df = pd.read_csv(file_path, skiprows=1, names=['Gmt_time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Convert GMT time to pandas datetime
    df['Gmt_time'] = pd.to_datetime(df['Gmt_time'], format='%d.%m.%Y %H:%M:%S.%f')
    
    # Convert GMT to CDT (GMT-5)
    df['Cdt_time'] = df['Gmt_time'] - timedelta(hours=5)
    
    # Extract hour and minute for easier filtering
    df['hour'] = df['Cdt_time'].dt.hour
    df['minute'] = df['Cdt_time'].dt.minute
    
    # Filter out zero volume records (placeholder data)
    df = df[df['Volume'] > 0]
    
    return df

def identify_cycles(df):
    """Identify main cycles and their second subcycles."""
    # Define main cycle hours and minutes
    main_cycle_times = [
        (1, 4), (5, 4), (9, 4), (13, 4), (17, 4), (21, 4)
    ]
    
    # Initialize results dictionary
    cycle_data = {}
    
    for hour, minute in main_cycle_times:
        # Filter main cycles
        main_cycles = df[(df['hour'] == hour) & (df['minute'] == minute)]
        
        # Calculate second subcycle time (40 minutes later)
        second_hour = hour
        second_minute = minute + 40
        if second_minute >= 60:
            second_hour = (second_hour + 1) % 24
            second_minute = second_minute - 60
        
        # Filter second subcycles
        second_cycles = df[(df['hour'] == second_hour) & (df['minute'] == second_minute)]
        
        # Store in dictionary
        cycle_name = f"{hour:02d}:{minute:02d} CDT"
        cycle_data[cycle_name] = {
            'main_cycles': main_cycles,
            'second_cycles': second_cycles,
            'second_time': f"{second_hour:02d}:{second_minute:02d} CDT"
        }
    
    return cycle_data

def analyze_price_returns(cycle_data, df):
    """Analyze if price returns to main cycle opening price during the second subcycle."""
    results = []
    
    for cycle_name, data in cycle_data.items():
        main_cycles = data['main_cycles']
        second_time = data['second_time']
        
        # Match main cycles with their corresponding second subcycles
        for _, main_row in main_cycles.iterrows():
            main_date = main_row['Cdt_time'].date()
            main_open = main_row['Open']
            main_time = main_row['Cdt_time']
            
            # Calculate second subcycle time range (40-80 minutes after main cycle start)
            second_start = main_time + timedelta(minutes=40)
            second_end = main_time + timedelta(minutes=80)
            
            # Get all price data during the second subcycle
            second_cycle_data = df[(df['Cdt_time'] >= second_start) & 
                                  (df['Cdt_time'] < second_end) & 
                                  (df['Cdt_time'].dt.date == main_date)]
            
            if not second_cycle_data.empty:
                # Check if any price during the second subcycle is within 0.1% of main cycle open
                price_returned = False
                closest_price = None
                min_diff_pct = float('inf')
                
                for _, row in second_cycle_data.iterrows():
                    # Check all price points (open, high, low, close)
                    for price_type in ['Open', 'High', 'Low', 'Close']:
                        price = row[price_type]
                        diff_pct = abs((price - main_open) / main_open * 100)
                        
                        if diff_pct < abs(min_diff_pct):
                            min_diff_pct = (price - main_open) / main_open * 100
                            closest_price = price
                        
                        if diff_pct <= 0.1:
                            price_returned = True
                
                # Add to results
                results.append({
                    'Date': main_date,
                    'Main_Cycle': cycle_name,
                    'Main_Open': main_open,
                    'Second_Cycle': second_time,
                    'Closest_Price': closest_price,
                    'Diff_Pct': min_diff_pct,
                    'Price_Returned': price_returned
                })
    
    return pd.DataFrame(results)

def generate_statistics(results_df):
    """Generate statistics from the results."""
    stats = {}
    
    # Overall statistics
    total_cycles = len(results_df)
    within_threshold = results_df['Price_Returned'].sum()
    pct_within_threshold = (within_threshold / total_cycles) * 100 if total_cycles > 0 else 0
    
    stats['overall'] = {
        'Total_Cycles': total_cycles,
        'Within_0.1_Pct': within_threshold,
        'Pct_Within_Threshold': pct_within_threshold,
        'Avg_Deviation': results_df['Diff_Pct'].abs().mean(),
        'Max_Positive_Dev': results_df['Diff_Pct'].max(),
        'Max_Negative_Dev': results_df['Diff_Pct'].min()
    }
    
    # Statistics by cycle time
    for cycle_name in results_df['Main_Cycle'].unique():
        cycle_df = results_df[results_df['Main_Cycle'] == cycle_name]
        
        cycle_total = len(cycle_df)
        cycle_within = cycle_df['Within_0.1_Pct'].sum()
        cycle_pct = (cycle_within / cycle_total) * 100 if cycle_total > 0 else 0
        
        stats[cycle_name] = {
            'Total_Cycles': cycle_total,
            'Within_0.1_Pct': cycle_within,
            'Pct_Within_Threshold': cycle_pct,
            'Avg_Deviation': cycle_df['Diff_Pct'].abs().mean(),
            'Max_Positive_Dev': cycle_df['Diff_Pct'].max(),
            'Max_Negative_Dev': cycle_df['Diff_Pct'].min()
        }
    
    return stats

def main():
    # Directory containing CSV files
    data_dir = r"c:\sqlite\CME-NQ"
    
    # Load all CSV files
    all_data = pd.DataFrame()
    for file_path in glob.glob(os.path.join(data_dir, "USATECH.IDXUSD_Candlestick_1_M_BID_*.csv")):
        try:
            df = load_and_prepare_data(file_path)
            all_data = pd.concat([all_data, df])
            print(f"Loaded {file_path}, records: {len(df)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Remove duplicates (in case of overlapping data files)
    all_data = all_data.drop_duplicates(subset=['Gmt_time'])
    
    # Sort by time
    all_data = all_data.sort_values('Gmt_time')
    
    print(f"Total records after processing: {len(all_data)}")
    print(f"Date range: {all_data['Cdt_time'].min()} to {all_data['Cdt_time'].max()}")
    
    # Count unique trading days
    trading_days = all_data['Cdt_time'].dt.date.nunique()
    print(f"Number of trading days: {trading_days}")
    
    # Identify cycles
    cycle_data = identify_cycles(all_data)
    
    # Analyze price returns
    results_df = analyze_price_returns(cycle_data, all_data)  # Pass all_data as second parameter
    
    # Generate statistics
    stats = generate_statistics(results_df)
    
    # Print results by cycle
    print("\n--- Results by Cycle ---")
    for cycle_name in sorted(results_df['Main_Cycle'].unique()):
        cycle_df = results_df[results_df['Main_Cycle'] == cycle_name]
        
        print(f"\n{cycle_name} Cycle:")
        print(f"{'Date':<12} {'Main Open':<12} {'Closest Price':<12} {'Diff %':<10} {'Returned to Open'}")
        print("-" * 70)
        
        for _, row in cycle_df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            print(f"{date_str:<12} {row['Main_Open']:<12.2f} {row['Closest_Price']:<12.2f} {row['Diff_Pct']:<10.3f} {'Yes' if row['Price_Returned'] else 'No'}")
    
    # Print overall statistics
    print("\n--- Overall Statistics ---")
    overall = stats['overall']
    print(f"Total cycles analyzed: {overall['Total_Cycles']}")
    print(f"Cycles with price return within 0.1%: {overall['Within_0.1_Pct']}/{overall['Total_Cycles']} ({overall['Pct_Within_Threshold']:.1f}%)")
    print(f"Average deviation from opening price: {overall['Avg_Deviation']:.3f}%")
    print(f"Largest positive deviation: {overall['Max_Positive_Dev']:.3f}%")
    print(f"Largest negative deviation: {overall['Max_Negative_Dev']:.3f}%")
    
    # Print statistics by cycle time
    print("\n--- Statistics by Cycle Time ---")
    for cycle_name in sorted([k for k in stats.keys() if k != 'overall']):
        cycle_stats = stats[cycle_name]
        print(f"\n{cycle_name} Cycle:")
        print(f"Cycles analyzed: {cycle_stats['Total_Cycles']}")
        print(f"Within 0.1%: {cycle_stats['Within_0.1_Pct']}/{cycle_stats['Total_Cycles']} ({cycle_stats['Pct_Within_Threshold']:.1f}%)")
        print(f"Average deviation: {cycle_stats['Avg_Deviation']:.3f}%")
        print(f"Max positive: {cycle_stats['Max_Positive_Dev']:.3f}%, Max negative: {cycle_stats['Max_Negative_Dev']:.3f}%")

if __name__ == "__main__":
    main()