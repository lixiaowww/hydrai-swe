import sqlite3
import random
from datetime import datetime, timedelta
import os

DB_FILE = "swe_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS swe_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            swe_mm REAL NOT NULL,
            data_source TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON swe_data(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_source ON swe_data(data_source)')
    
    # Check if data exists
    cursor.execute("SELECT COUNT(*) FROM swe_data")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("Database empty. Generating realistic historical data...")
        
        # Generate 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Simulate seasonal SWE (peak in March, low in summer)
            month = current_date.month
            day_of_year = current_date.timetuple().tm_yday
            
            # Base SWE curve
            if 1 <= month <= 4: # Winter/Spring
                base_swe = 40 + (day_of_year * 0.5) if month < 3 else 100 - ((day_of_year - 60) * 1.5)
            elif 11 <= month <= 12: # Early Winter
                base_swe = (day_of_year - 300) * 0.5
            else: # Summer/Fall
                base_swe = 0
                
            # Add noise and ensure non-negative
            swe_val = max(0, base_swe + random.uniform(-5, 5))
            
            # Add record
            data.append((
                current_date.strftime('%Y-%m-%d'),
                round(swe_val, 2),
                "simulated_station_network"
            ))
            
            current_date += timedelta(days=1)
            
        cursor.executemany("INSERT INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)", data)
        conn.commit()
        print(f"Inserted {len(data)} records.")
    else:
        print(f"Database already contains {count} records.")
        
    conn.close()

if __name__ == "__main__":
    init_db()
