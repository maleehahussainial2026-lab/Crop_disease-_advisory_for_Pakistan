# ============================================================
# ingest.py
# JOB: Read all 64 JSON files from dataset/ folder and
#      save them into PostgreSQL documents table
#      with their 384-dimensional embedding vectors
#
# ⚠️  RUN THIS ONLY ONCE before starting the server
#      Command: python ingest.py
# ============================================================

# json lets us read and parse .json files
import json

# os lets us work with file paths and folders
import os

# glob finds all files matching a pattern in a folder
# we use it to find all *.json files in dataset/ folder
import glob

# text() lets us write raw SQL queries as strings
from sqlalchemy import text

# get our database session from database.py
from database import SessionLocal

# get our embedding function from embeddings.py
# this converts chunk_text into 384 numbers
from embeddings import get_embedding


# ── Step 1: main ingest function ────────────────────────────
def ingest():

    # open a database session
    # think of this as opening a working connection to PostgreSQL
    db = SessionLocal()

    # ── Step 2: find all JSON files in dataset/ folder ──────
    # glob.glob searches for files matching the pattern
    # dataset/*.json means: all files ending in .json
    #                       inside the dataset/ folder
    json_files = glob.glob("dataset/*.json")

    # if no files found stop early and warn
    if not json_files:
        print("❌ No JSON files found in dataset/ folder")
        print("   Make sure your 64 JSON files are in backend/dataset/")
        return

    print(f"📂 Found {len(json_files)} JSON files to ingest")
    print("⏳ Starting ingestion — this may take 2 to 3 minutes...")
    print("   (MiniLM model runs 64 times to embed all chunks)\n")

    # ── Step 3: track counts for final report ───────────────
    success_count = 0       # how many documents inserted successfully
    skip_count = 0          # how many documents skipped (duplicates)
    error_count = 0         # how many documents had errors

    # ── Step 4: loop through every JSON file ────────────────
    for file_path in json_files:

        try:
            # open and read the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                doc = json.load(f)

            # get the chunk_id from the document
            # chunk_id is unique identifier like "wheat_yellow_rust_A"
            chunk_id = doc.get("chunk_id", "")

            # safety check: skip if chunk_id is missing
            if not chunk_id:
                print(f"⚠️  Skipping {file_path} — missing chunk_id")
                skip_count += 1
                continue

            # ── Step 5: check if document already exists ────
            # this prevents duplicate entries if you run
            # ingest.py more than once by mistake
            existing = db.execute(
                text("SELECT id FROM documents WHERE chunk_id = :chunk_id"),
                {"chunk_id": chunk_id}
            ).fetchone()

            if existing:
                print(f"⏭️  Skipping {chunk_id} — already in database")
                skip_count += 1
                continue

            # ── Step 6: generate embedding for chunk_text ───
            # FIX 2: Concatenate all searchable fields before embedding.
            # chunk_text alone is English-heavy and misses Urdu names,
            # symptom descriptions, and local terms used by farmers.
            # Including urdu_name, symptoms, cause, and favorable_conditions
            # dramatically improves recall for mixed Urdu/Punjabi queries.
            chunk_text = doc.get("chunk_text", "")
            urdu_name  = doc.get("urdu_name", "") or ""
            symptoms   = doc.get("symptoms", "") or ""
            cause      = doc.get("cause", "") or ""
            fav_cond   = doc.get("favorable_conditions", "") or ""
            embedding_text = " ".join(filter(None, [
                chunk_text, urdu_name, symptoms, cause, fav_cond
            ]))
            embedding = get_embedding(embedding_text)

            # ── Step 7: extract all fields from JSON ────────
            # we read every field from the JSON document
            # and prepare it for insertion into PostgreSQL

            # handle resistant_varieties — stored as TEXT[] array
            # some files may have this as list, ensure it is a list
            resistant_varieties = doc.get("resistant_varieties", [])
            if isinstance(resistant_varieties, str):
                resistant_varieties = [resistant_varieties]

            # handle province — stored as TEXT[] array
            province = doc.get("province", [])

            # handle season — stored as TEXT[] array
            season = doc.get("season", [])

            # handle source — stored as TEXT[] array
            source = doc.get("source", [])

            # ── Step 8: insert document into PostgreSQL ──────
            # we use raw SQL INSERT for simplicity
            # :field_name syntax safely passes values
            # preventing SQL injection attacks
            db.execute(text("""
                INSERT INTO documents (
                    chunk_id,
                    chunk_type,
                    crop,
                    crop_type,
                    disease_name,
                    urdu_name,
                    cause,
                    symptoms,
                    favorable_conditions,
                    crop_stage_affected,
                    yield_loss,
                    chemical_control,
                    application_timing,
                    resistant_varieties,
                    ipm_tips,
                    province,
                    season,
                    source,
                    image_file,
                    chunk_text,
                    embedding
                ) VALUES (
                    :chunk_id,
                    :chunk_type,
                    :crop,
                    :crop_type,
                    :disease_name,
                    :urdu_name,
                    :cause,
                    :symptoms,
                    :favorable_conditions,
                    :crop_stage_affected,
                    :yield_loss,
                    :chemical_control,
                    :application_timing,
                    :resistant_varieties,
                    :ipm_tips,
                    :province,
                    :season,
                    :source,
                    :image_file,
                    :chunk_text,
                    :embedding
                )
            """), {
                "chunk_id"              : chunk_id,
                "chunk_type"            : doc.get("chunk_type", ""),
                "crop"                  : doc.get("crop", ""),
                "crop_type"             : doc.get("crop_type", ""),
                "disease_name"          : doc.get("disease_name", ""),
                "urdu_name"             : doc.get("urdu_name", ""),
                "cause"                 : doc.get("cause", ""),
                "symptoms"              : doc.get("symptoms", ""),
                "favorable_conditions"  : doc.get("favorable_conditions", ""),
                "crop_stage_affected"   : doc.get("crop_stage_affected", ""),
                "yield_loss"            : doc.get("yield_loss", ""),
                "chemical_control"      : doc.get("chemical_control", ""),
                "application_timing"    : doc.get("application_timing", ""),
                "resistant_varieties"   : resistant_varieties,
                "ipm_tips"              : doc.get("ipm_tips", ""),
                "province"              : province,
                "season"                : season,
                "source"                : source,
                "image_file"            : doc.get("image_file", ""),
                "chunk_text"            : chunk_text,
                # FIX 3: Store embedding as JSON string (TEXT column).
                # json.dumps ensures retrieval.py's json.loads() parses
                # it correctly on all PostgreSQL versions.
                "embedding"             : json.dumps(embedding)
            })

            # print progress for each successful insert
            print(f"✅ Ingested: {chunk_id}")
            success_count += 1

        except Exception as e:
            # if any single file fails print the error
            # but continue with remaining files
            print(f"❌ Error ingesting {file_path}: {e}")
            # ⚠️ IMPORTANT: rollback the broken transaction so
            # the next file can start fresh. Without this, one
            # error puts PostgreSQL in a failed state and every
            # remaining file also fails ("transaction aborted").
            db.rollback()
            error_count += 1
            continue

    # ── Step 9: commit all inserts to database ───────────────
    # commit() is like pressing SAVE
    # without this nothing is actually written to PostgreSQL
    db.commit()

    # close the database session when done
    db.close()

    # ── Step 10: print final summary report ──────────────────
    print("\n" + "="*50)
    print("📊 INGESTION COMPLETE — SUMMARY")
    print("="*50)
    print(f"✅ Successfully ingested : {success_count} documents")
    print(f"⏭️  Skipped (duplicates)  : {skip_count} documents")
    print(f"❌ Errors                : {error_count} documents")
    print(f"📦 Total files processed : {len(json_files)}")
    print("="*50)
    print("\n🎉 Your RAG dataset is now in PostgreSQL!")
    print("   You can now start the FastAPI server with:")
    print("   uvicorn main:app --reload")


# ── run ingest when file is executed directly ────────────────
# python ingest.py
if __name__ == "__main__":
    ingest()