# ============================================================
# retrieval.py
# FIX: Always pair identification + management chunks so that
#      Treatment and IPM Tips are never missing from LLM context.
# ============================================================

import json
import math
from sqlalchemy import text
from database import SessionLocal
from embeddings import get_embedding
from query_processor import preprocess_query  # FIX 4


def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def retrieve(
    query: str,
    crop: str = None,
    province: str = None,
    top_k: int = 3
) -> list:

    db = SessionLocal()

    try:
        print(f"🔍 Retrieving for query: '{query}'")
        # FIX 4: Enrich query with synonyms, spell correction, and stopword removal
        enriched_query = preprocess_query(query)
        query_embedding = get_embedding(enriched_query)

        # Build SQL with optional filters
        base_sql = """
            SELECT
                chunk_id, chunk_type, crop, disease_name, urdu_name,
                cause, symptoms, favorable_conditions, crop_stage_affected,
                yield_loss, chemical_control, application_timing,
                resistant_varieties, ipm_tips, province, season,
                source, image_file, chunk_text, embedding
            FROM documents
            WHERE 1=1
        """
        params = {}

        if crop and crop.strip():
            base_sql += " AND LOWER(crop) = LOWER(:crop)"
            params["crop"] = crop.strip()
            print(f"   🌾 Filtering by crop: {crop}")

        if province and province.strip():
            base_sql += " AND province @> ARRAY[:province]"
            params["province"] = province.strip()
            print(f"   📍 Filtering by province: {province}")

        results = db.execute(text(base_sql), params).fetchall()

        # If filters return nothing, retry without filters
        if not results and (crop or province):
            print("   ⚠️  No results with filters — retrying without filters")
            results = db.execute(text("""
                SELECT chunk_id, chunk_type, crop, disease_name, urdu_name,
                       cause, symptoms, favorable_conditions, crop_stage_affected,
                       yield_loss, chemical_control, application_timing,
                       resistant_varieties, ipm_tips, province, season,
                       source, image_file, chunk_text, embedding
                FROM documents
            """), {}).fetchall()

        # Build a lookup map: chunk_id -> row (for later pairing)
        all_rows_by_id = {row.chunk_id: row for row in results}

        # Calculate cosine similarity in Python
        scored = []
        for row in results:
            try:
                doc_embedding = json.loads(row.embedding)
                score = cosine_similarity(query_embedding, doc_embedding)
                scored.append((score, row))
            except Exception:
                continue

        # Sort by score descending and take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = scored[:top_k]

        # ── PAIRING FIX ──────────────────────────────────────────────
        # For disease chunk types (identification / management), always
        # ensure the partner chunk is included so the LLM has complete
        # cause + treatment + IPM info in one context.
        #
        # Convention in the dataset: chunk_ids end with _A or _B.
        #   _A  = identification  (cause, symptoms — NO treatment/IPM)
        #   _B  = management      (treatment, IPM   — NO cause/symptoms)
        #
        # Strategy: after scoring top_k by similarity, scan for disease
        # chunks and inject their missing partner at the same confidence.
        DISEASE_TYPES = {"identification", "management"}
        paired_ids = set()
        extra_chunks = []

        for score, row in top_results:
            if row.chunk_type not in DISEASE_TYPES:
                continue
            chunk_id = row.chunk_id
            if chunk_id.endswith("_A"):
                partner_id = chunk_id[:-2] + "_B"
            elif chunk_id.endswith("_B"):
                partner_id = chunk_id[:-2] + "_A"
            else:
                continue

            # Only add partner if not already in top_results and not already added
            top_ids = {r.chunk_id for _, r in top_results}
            if partner_id not in top_ids and partner_id not in paired_ids:
                partner_row = all_rows_by_id.get(partner_id)
                if partner_row:
                    extra_chunks.append((score, partner_row))
                    paired_ids.add(partner_id)
                    print(f"   🔗 Paired partner chunk: {partner_id}")

        # Merge top results with their partners (partners inherit the same score)
        merged = top_results + extra_chunks

        # Re-sort merged list by score so identification comes before management
        merged.sort(key=lambda x: x[0], reverse=True)

        # ── Format results ────────────────────────────────────────────
        formatted_results = []
        for score, row in merged:
            if score >= 0.85:
                confidence = "High"
            elif score >= 0.65:
                confidence = "Medium"
            elif score >= 0.30:
                confidence = "Low — related topic found"
            else:
                confidence = "Very Low — please verify with local agricultural officer"

            formatted_results.append({
                "chunk_id"             : row.chunk_id,
                "chunk_type"           : row.chunk_type,
                "crop"                 : row.crop,
                "disease_name"         : row.disease_name,
                "urdu_name"            : row.urdu_name,
                "cause"                : row.cause,
                "symptoms"             : row.symptoms,
                "favorable_conditions" : row.favorable_conditions,
                "crop_stage_affected"  : row.crop_stage_affected,
                "yield_loss"           : row.yield_loss,
                "chemical_control"     : row.chemical_control,
                "application_timing"   : row.application_timing,
                "resistant_varieties"  : row.resistant_varieties,
                "ipm_tips"             : row.ipm_tips,
                "province"             : row.province,
                "season"               : row.season,
                "source"               : row.source,
                "image_file"           : row.image_file,
                "chunk_text"           : row.chunk_text,
                "similarity_score"     : round(score, 4),
                "confidence"           : confidence
            })

            print(f"   📄 Found: {row.disease_name} [{row.chunk_type}] "
                  f"(score: {round(score, 4)}, confidence: {confidence})")

        print(f"   ✅ Returning {len(formatted_results)} chunks\n")
        return formatted_results

    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

    finally:
        db.close()


def test_retrieval():
    print("=" * 50)
    print("TEST 1: Query with crop filter")
    print("=" * 50)
    results = retrieve(
        query="yellow stripes on wheat leaves",
        crop="wheat",
        province="Punjab"
    )
    for r in results:
        print(f"  Disease   : {r['disease_name']} [{r['chunk_type']}]")
        print(f"  Score     : {r['similarity_score']}")
        print(f"  Treatment : {r['chemical_control']}")
        print(f"  IPM Tips  : {r['ipm_tips']}")
        print()

    print("=" * 50)
    print("TEST 2: Cotton leaf curl")
    print("=" * 50)
    results = retrieve(query="cotton leaves are curling")
    for r in results:
        print(f"  Disease   : {r['disease_name']} [{r['chunk_type']}]")
        print(f"  Score     : {r['similarity_score']}")
        print(f"  Treatment : {r['chemical_control']}")
        print(f"  IPM Tips  : {r['ipm_tips']}")
        print()


if __name__ == "__main__":
    test_retrieval()