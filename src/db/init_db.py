"""
init_db.py - create a simple SQLite DB to store logs and predictions
Run: python src/db/init_db.py
"""
import sqlite3, os, datetime
DB="src/db/smartwaste.db"
os.makedirs(os.path.dirname(DB), exist_ok=True)
conn=sqlite3.connect(DB)
c=conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    filename TEXT,
    predicted_class TEXT,
    confidence REAL
)""")
conn.commit(); conn.close()
print("DB initialized at", DB)
