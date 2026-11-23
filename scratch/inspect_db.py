import sqlite3

conn = sqlite3.connect('showroom.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

for t in tables:
    tname = t[0]
    cursor.execute(f"PRAGMA table_info({tname});")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"Table {tname}: {cols}")
    cursor.execute(f"SELECT * FROM {tname} LIMIT 5;")
    rows = cursor.fetchall()
    print(f"  Rows count: {len(rows)}")
