# ============================================================
# database.py
# JOB: Connect to PostgreSQL database
# All other files import get_db() from here to use database
# ============================================================

# python-dotenv lets us read secret values from .env file
# so we never hardcode passwords in our code
from dotenv import load_dotenv

# SQLAlchemy is a Python library that talks to PostgreSQL
# create_engine = creates the actual database connection
# text = lets us write raw SQL queries as strings
from sqlalchemy import create_engine, text

# sessionmaker = creates a session (like a working space)
# where we can run queries and save data
from sqlalchemy.orm import sessionmaker

# os lets us read environment variables like DATABASE_URL
import os

# ── Step 1: Load values from .env file into memory ──────────
# After this line os.getenv() can read DATABASE_URL and GROQ_API_KEY
load_dotenv()

# ── Step 2: Read the database URL from .env file ────────────
# DATABASE_URL looks like:
# postgresql://postgres:yourpassword@localhost:5432/crop_disease_db
DATABASE_URL = os.getenv("DATABASE_URL")

# ── Step 3: Create the database engine ──────────────────────
# engine is the actual connection to PostgreSQL
# think of it as the cable connecting Python to your database
engine = create_engine(DATABASE_URL)

# ── Step 4: Create a session factory ────────────────────────
# SessionLocal is a class that creates new sessions on demand
# autocommit=False means we manually confirm (commit) saves
# autoflush=False means we control when data is written
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine           # attach session to our engine above
)

# ── Step 5: Test function to confirm connection works ────────
# Call this once manually to confirm database is reachable
def test_connection():
    try:
        # open a direct connection from the engine
        with engine.connect() as conn:
            # run the simplest possible SQL query
            conn.execute(text("SELECT 1"))
        print("✅ Database connected successfully")
    except Exception as e:
        # if connection fails print the error clearly
        print(f"❌ Database connection failed: {e}")

# ── Step 6: get_db function ──────────────────────────────────
# This is the function ALL other files will import and use
# It opens a session, gives it to the calling file,
# then automatically closes it when done
# 'yield' means: give this session to whoever asked for it
# the code after yield runs automatically when they are done
def get_db():
    db = SessionLocal()       # open a new session
    try:
        yield db              # hand the session to the caller
    finally:
        db.close()            # always close session when done
                              # even if an error occurred


# ── Step 7: Run test when file is executed directly ─────────
# This block only runs if you do: python database.py
# It will NOT run when other files import from this file
if __name__ == "__main__":
    test_connection()
