import sqlite3
from config import DATABASE_PATH

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        compound TEXT,
        smiles TEXT,
        vina_score REAL,
        cnn_score REAL,
        tox_score REAL,
        composite_score REAL,
        md_rmsd REAL,
        target TEXT,
        library TEXT,
        run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def insert_result(data):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    c.execute("""
    INSERT INTO results
    (compound, smiles, vina_score, cnn_score, tox_score,
     composite_score, md_rmsd, target, library)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()
