# ============================================================
# setup_db.py
# JOB: Create all tables in PostgreSQL from Python
# RUN THIS ONCE before running ingest.py
# Command: python setup_db.py
# ============================================================

from sqlalchemy import text
from database import engine

def setup():
    with engine.connect() as conn:

        print("⏳ Setting up database tables...")


        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("✅ pgvector extension ready")

        # Step 2: Drop old tables if they exist (clean slate)
        conn.execute(text("DROP TABLE IF EXISTS retrieved_chunks CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS chat_logs CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS documents CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS users CASCADE;"))
        print("✅ Old tables dropped")

        # Step 3: Create documents table (main RAG knowledge base)
        conn.execute(text("""
            CREATE TABLE documents (
                id                   SERIAL PRIMARY KEY,
                chunk_id             TEXT UNIQUE NOT NULL,
                chunk_type           TEXT,
                crop                 TEXT,
                crop_type            TEXT,
                disease_name         TEXT,
                urdu_name            TEXT,
                cause                TEXT,
                symptoms             TEXT,
                favorable_conditions TEXT,
                crop_stage_affected  TEXT,
                yield_loss           TEXT,
                chemical_control     TEXT,
                application_timing   TEXT,
                resistant_varieties  TEXT[],
                ipm_tips             TEXT,
                province             TEXT[],
                season               TEXT[],
                source               TEXT[],
                image_file           TEXT,
                chunk_text           TEXT,
                embedding            TEXT
            );
        """))
        print("✅ documents table created")

        # Step 4: Create users table
        conn.execute(text("""
            CREATE TABLE users (
                id                 SERIAL PRIMARY KEY,
                username           TEXT UNIQUE NOT NULL,
                email              TEXT UNIQUE NOT NULL,
                hashed_password    TEXT NOT NULL,
                data_consent       BOOLEAN DEFAULT FALSE,
                consent_timestamp  TIMESTAMP,
                preferred_language TEXT DEFAULT 'english',
                created_at         TIMESTAMP DEFAULT NOW()
            );
        """))
        print("✅ users table created")

        # Step 5: Create chat_logs table
        conn.execute(text("""
            CREATE TABLE chat_logs (
                id                SERIAL PRIMARY KEY,
                user_id           INTEGER REFERENCES users(id) ON DELETE SET NULL,
                session_id        TEXT,
                query_text        TEXT,
                response_text     TEXT,
                crop_selected     TEXT,
                province_selected TEXT,
                language_used     TEXT DEFAULT 'english',
                timestamp         TIMESTAMP DEFAULT NOW()
            );
        """))
        print("✅ chat_logs table created")

        # Step 6: Create retrieved_chunks table
        conn.execute(text("""
            CREATE TABLE retrieved_chunks (
                id               SERIAL PRIMARY KEY,
                chat_log_id      INTEGER REFERENCES chat_logs(id) ON DELETE CASCADE,
                chunk_text       TEXT,
                disease_name     TEXT,
                similarity_score FLOAT,
                source           TEXT,
                image_file       TEXT
            );
        """))
        print("✅ retrieved_chunks table created")

        conn.commit()

        print("\n🎉 All tables created successfully!")
        print("   Now run: python ingest.py")

if __name__ == "__main__":
    setup()