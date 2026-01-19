import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(num_records=2000):
    # Simulating the header from the screenshot:
    # DATE, line_description, bde_order_number, GCAS, GCAS_desc, cell_description, counter_type_description, count
    
    lines = ['[Sonos] Handle Assembly 1', '[Sonos] Handle Assembly 2', 'Main Assembly Line A']
    
    # Generate dates mostly in late 2025 like the screenshot
    start_date = datetime(2025, 9, 1)
    dates = [(start_date + timedelta(days=random.randint(0, 150))).strftime('%Y-%m-%d') for _ in range(num_records)]
    
    # P-style orders
    orders = [f'P{random.randint(920000000, 920999999)}' for _ in range(30)]
    
    gcas_map = {
        '90272164': 'OP021 cpl BK G5 5 Mode MN/NA MHF',
        '20240464': 'OP030 toothbrush cpl Black G2 generic MHF',
        '90361524': 'OP021 cpl BP G3 3 Mode MN/NA MHF',
        '91814556': 'OP021 toothbrush cpl G5 BK/BK',
        '20125655': 'OP030 cpl LT PK G2 MN/NA MHF',
    }
    gcas_keys = list(gcas_map.keys())
    
    cell_desc = "C12: Manual Inspection / Unloading"
    
    # Counter types including user's specific "Schlechtteil" and original "Hand written"
    # To test the app flexibility, we'll mix them.
    counter_types = [
        '0 - Ungültig - Schlechtteil', 
        'Good_In from Line', 
        'Hand written error type A', 
        'Machine Reject',
        'System Log'
    ]
    
    data = []
    
    for i in range(num_records):
        date = dates[i]
        line = random.choice(lines)
        order = random.choice(orders)
        gcas_code = random.choice(gcas_keys)
        gcas_desc = gcas_map[gcas_code]
        
        # Weighted types
        # 60% Good, 10% Ungültig, 10% Hand written, etc.
        c_type = random.choices(counter_types, weights=[0.1, 0.7, 0.1, 0.05, 0.05], k=1)[0]
        
        count = 0
        if c_type == 'Good_In from Line':
            count = random.randint(1, 10)
        elif c_type == '0 - Ungültig - Schlechtteil':
            # Screenshot shows lots of 0s, some 1s, 2s
            count = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]
        else:
            count = random.randint(0, 3)
            
        data.append([date, line, order, gcas_code, gcas_desc, cell_desc, c_type, count])
        
    df = pd.DataFrame(data, columns=[
        'DATE', 'line_description', 'bde_order_number', 'GCAS', 'GCAS_desc', 
        'cell_description', 'counter_type_description', 'count'
    ])
    return df

if __name__ == "__main__":
    print("Generating updated dummy data matching user screenshot...")
    df = generate_data(2000)
    print(f"Generated {len(df)} records.")
    output_file = "dummy_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
