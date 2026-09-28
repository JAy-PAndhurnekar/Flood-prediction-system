import os
import sqlite3
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "database.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    location TEXT,
    rainfall REAL,
    temperature REAL,
    humidity REAL,
    river_level REAL,
    soil_moisture REAL,
    previous_rainfall REAL,
    flood_probability REAL,
    risk_level TEXT,
    model_confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location TEXT,
    risk_level TEXT,
    probability REAL,
    message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", ("admin@floodai.local",))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
            ("Admin", "admin@floodai.local", generate_password_hash("admin123"), "admin")
        )
    conn.commit()
    conn.close()
