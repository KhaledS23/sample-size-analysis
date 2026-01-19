import pandas as pd

def verify():
    print("Loading data...")
    df = pd.read_csv("dummy_data.csv")
    
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Identify Hand Written Defects (Case insensitive starts with "Hand written")
    # We create a new column for easier grouping
    df['is_defect'] = df['counter_type_description'].str.strip().str.lower().str.startswith("hand written")
    
    # Simulate user selection (picking "Good_In from Line")
    good_parts_selection = ["Good_In from Line"]
    df['is_good'] = df['counter_type_description'].isin(good_parts_selection)
    
    print(f"Total Rows: {len(df)}")
    print(f"Defect Rows: {df['is_defect'].sum()}")
    print(f"Good Rows: {df['is_good'].sum()}")
    
    if df['is_defect'].sum() == 0:
        print("WARNING: No defects found using filter 'starts with Hand written'")
    if df['is_good'].sum() == 0:
        print("WARNING: No good parts found using selection 'Good_In from Line'")
        
    # Aggregation Helper
    def calculate_ppm(grouped_df):
        defect_counts = grouped_df[grouped_df['is_defect']]['count'].sum()
        good_counts = grouped_df[grouped_df['is_good']]['count'].sum()
        total_parts = good_counts + defect_counts
        if total_parts == 0:
            return 0.0
        return (defect_counts / total_parts) * 1_000_000

    # Check Global PPM
    global_ppm = calculate_ppm(df)
    print(f"Global PPM: {global_ppm:.2f}")

    # Check Top Offender GCAS
    gcas_groups = df.groupby('GCAS')
    gcas_ppm_data = []
    for gcas_id, group in gcas_groups:
        ppm = calculate_ppm(group)
        gcas_ppm_data.append({'GCAS': gcas_id, 'ppm': ppm})
        
    df_gcas = pd.DataFrame(gcas_ppm_data).sort_values('ppm', ascending=False)
    print("\nTop 5 GCAS Offenders:")
    print(df_gcas.head(5))
    
    # Check Orders
    print("\nOrder Stats Sample:")
    order_groups = df.groupby('bde_order_number')
    order_ppm_data = []
    for order_id, group in order_groups:
        ppm = calculate_ppm(group)
        order_ppm_data.append(ppm)
        
    print(f"Max Order PPM: {max(order_ppm_data):.2f}")
    print(f"Min Order PPM: {min(order_ppm_data):.2f}")

if __name__ == "__main__":
    verify()
