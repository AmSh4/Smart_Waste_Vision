"""
log_prediction.py - helper to insert prediction records into SQLite DB
"""
import sqlite3, os, datetime

DB="src/db/smartwaste.db"

def log_prediction(filename, predicted_class, confidence):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn=sqlite3.connect(DB)
    c=conn.cursor()
    c.execute("INSERT INTO predictions (timestamp, filename, predicted_class, confidence) VALUES (?,?,?,?)",
              (datetime.datetime.utcnow().isoformat(), filename, predicted_class, float(confidence)))
    conn.commit(); conn.close()
