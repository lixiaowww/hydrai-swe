import sqlite3
from datetime import datetime, timedelta
import random

DB_FILE = "swe_data.db"

def seed_data():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if data exists
    cursor.execute("SELECT COUNT(*) FROM swe_data")
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"Database already has {count} records.")
        conn.close()
        return

    print("Seeding database with sample data...")
    
    # Generate data for the last 365 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = []
    current_date = start_date
    
    # Simulate a seasonal SWE curve
    while current_date <= end_date:
        month = current_date.month
        
        # Simple seasonal model: Peak in March (3), low in summer
        if 1 <= month <= 5: # Jan-May
            base_swe = 50 + (month * 20) # Increasing
        elif 6 <= month <= 10: # Jun-Oct
            base_swe = 0 # No snow
        else: # Nov-Dec
            base_swe = (month - 10) * 20 # Accumulating
            
        # Add noise
        swe = max(0, base_swe + random.uniform(-10, 10))
        
        data.append((
            current_date.strftime('%Y-%m-%d'),
            round(swe, 2),
            'synthetic_seed'
        ))
        
        current_date += timedelta(days=1)
        
    cursor.executemany(
        "INSERT INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)",
        data
    )
    
    conn.commit()
    print(f"Inserted {len(data)} records.")
    conn.close()

if __name__ == "__main__":
    seed_data()
