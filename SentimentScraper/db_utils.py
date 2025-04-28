import sqlite3
import os

def init_db():
    """Initialize SQLite database for storing sentiment analysis results."""
    try:
        # Use absolute path to ensure consistent database location
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DB_PATH = os.path.join(BASE_DIR, 'results.db')
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_results (
                run_id TEXT PRIMARY KEY,
                ticker TEXT,
                run_timestamp TEXT,
                headline TEXT,
                url TEXT,
                summary TEXT,
                sentiment_score REAL,
                source TEXT,
                published_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Database initialized successfully at {DB_PATH}")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        raise