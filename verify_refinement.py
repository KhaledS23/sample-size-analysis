import pandas as pd

def verify():
    print("Loading new dummy data...")
    try:
        df = pd.read_csv("dummy_data.csv")
    except FileNotFoundError:
        print("Error: dummy_data.csv not found. Please run generate_dummy_data.py first.")
        return

    print(f"Columns: {df.columns.tolist()}")
    
    # Check for new columns from screenshot
    required_cols = ['DATE', 'GCAS_desc', 'cell_description']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing expected columns: {missing}")
    else:
        print("Schema check passed.")

    # Simulate App Logic with Defect Filter = "0 - Ungültig" (as seen in screenshot)
    # Note: Logic uses startsWith, so we need the prefix
    defect_pattern = "0 - " 
    # Or test "Hand written" if that's what we want to verify. Let's verify both if mixed.
    # Current generator mixes them.
    
    # Test 1: Defect Pattern = "0 - " (representing Ungültig rows)
    print(f"\nTesting Defect Pattern: '{defect_pattern}'")
    df['is_defect'] = df['counter_type_description'].str.strip().str.lower().str.startswith(defect_pattern.lower())
    
    # Simulate user selection (picking "Good_In from Line")
    good_parts_selection = ["Good_In from Line"]
    df['is_good'] = df['counter_type_description'].isin(good_parts_selection)
    
    defect_count = df['is_defect'].sum()
    good_count = df['is_good'].sum()
    
    print(f"Defect Rows found: {defect_count}")
    print(f"Good Rows found: {good_count}")
    
    if defect_count == 0:
        print("WARNING: No defects found using filter. Check generator or logic.")
        
    # Test 2: Defect Pattern = "Hand written" (Original requirement)
    defect_pattern_2 = "Hand written"
    print(f"\nTesting Defect Pattern: '{defect_pattern_2}'")
    mask_2 = df['counter_type_description'].str.strip().str.lower().str.startswith(defect_pattern_2.lower())
    print(f"Defect Rows found: {mask_2.sum()}")


if __name__ == "__main__":
    verify()
