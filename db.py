import sqlite3
conn = sqlite3.connect("chat.db")
cursor = conn.cursor()
Create = """CREATE TABLE IF NOT EXISTS ChatTable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    message TEXT NOT NULL,
    dateTime DATETIME DEFAULT CURRENT_TIMESTAMP
)"""
cursor.execute(Create)