# ============================================================
# llm.py - Updated for small_talk + general_advisory support
# ============================================================

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LABELS = {
    "english": {
        "disease"      : "Disease Name",
        "cause"        : "Cause",
        "treatment"    : "Recommended Treatment",
        "ipm"          : "IPM Tips",
        "confidence"   : "Confidence Level",
        "evidence"     : "Evidence",
        "insufficient" : "Insufficient information in agricultural records"
    },
    "urdu": {
        "disease"      : "بیماری کا نام",
        "cause"        : "وجہ",
        "treatment"    : "تجویز کردہ علاج",
        "ipm"          : "آئی پی ایم تجاویز",
        "confidence"   : "اعتماد کی سطح",
        "evidence"     : "ثبوت",
        "insufficient" : "زرعی ریکارڈ میں ناکافی معلومات"
    }
}

# ── NEW: chunk types that are NOT disease records ─────────────
NON_DISEASE_TYPES = {"small_talk", "general_advisory"}


def safe_join(value):
    """Safely join a list or return string as-is"""
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    if isinstance(value, str):
        cleaned = value.strip("[]").replace("'", "").replace('"', '')
        return cleaned
    return str(value)


def build_context(chunks: list) -> str:
    if not chunks:
        return ""

    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        part = f"""
--- Retrieved Document {i} ---
Disease     : {chunk.get('disease_name', 'N/A')}
Crop        : {chunk.get('crop', 'N/A')}
Chunk Type  : {chunk.get('chunk_type', 'N/A')}
Cause       : {chunk.get('cause', 'N/A')}
Symptoms    : {chunk.get('symptoms', 'N/A')}
Treatment   : {chunk.get('chemical_control', 'N/A')}
IPM Tips    : {chunk.get('ipm_tips', 'N/A')}
Timing      : {chunk.get('application_timing', 'N/A')}
Varieties   : {safe_join(chunk.get('resistant_varieties'))}
Yield Loss  : {chunk.get('yield_loss', 'N/A')}
Sources     : {safe_join(chunk.get('source'))}
Similarity  : {chunk.get('similarity_score', 0)}
"""
        context_parts.append(part)

    return "\n".join(context_parts)


def build_system_prompt(language: str, confidence: str) -> str:
    lang = language.lower() if language else "english"

    if lang == "auto":
        language_instruction = (
            "Detect the language of the farmer's question "
            "and respond in that same language. "
            "If the question is in Urdu respond in Urdu. "
            "If in English respond in English."
        )
    elif lang == "urdu":
        language_instruction = (
            "Always respond in Urdu language only. "
            "Use Urdu script. Do not use English in your response "
            "except for scientific names of pathogens."
        )
    else:
        language_instruction = "Always respond in English language only."

    labels = LABELS.get(lang, LABELS["english"])

    system_prompt = f"""
You are an expert agricultural disease advisor for Pakistani farmers.
Your knowledge comes ONLY from the retrieved documents provided to you.

LANGUAGE INSTRUCTION:
{language_instruction}

CONVERSATION AWARENESS:
- You have access to the full conversation history above.
- When the farmer asks a follow-up question (e.g., "is it a disease?", "what about the tubers?", "how to treat it?"), you MUST refer to the context of the previous messages to give a relevant, connected answer.
- NEVER treat a follow-up as a brand new unrelated question.

CRITICAL RULES - FOLLOW EXACTLY:
1. Answer ONLY using information from the retrieved documents below.
2. ALWAYS fill in ALL fields completely - never leave any field empty.
3. For Treatment: copy the EXACT chemical treatment and dosage from documents.
4. For IPM Tips: copy the EXACT IPM tips from documents.
5. For Evidence: list the actual source names from documents.
6. NEVER write "Insufficient information" if documents contain ANY related disease info.
7. Even if the match is partial, provide the best available information from documents.
8. Only write insufficient if documents have absolutely zero relevant information.
9. YOU MUST RESPOND IN THE SAME LANGUAGE AS SPECIFIED. If Urdu, write everything in Urdu script.
10. When responding in Urdu, translate ALL field values into Urdu - do not leave English text.

RESPONSE FORMAT (fill ALL fields - no exceptions):
{labels['disease']}: [disease name]
{labels['cause']}: [cause - translate to response language]
{labels['treatment']}: [exact treatment with dosage - translate to response language]
{labels['ipm']}: [IPM tips - translate to response language]
{labels['confidence']}: {confidence}
{labels['evidence']}: [source names]
"""
    return system_prompt


# ── NEW: system prompt for small_talk chunk type ──────────────
def build_smalltalk_prompt(language: str) -> str:
    lang = language.lower() if language else "english"

    if lang == "urdu":
        language_instruction = (
            "Always respond in Urdu script only. "
            "Be warm, friendly and helpful. Do not use English."
        )
    else:
        language_instruction = (
            "Always respond in English only. "
            "Be warm, friendly and helpful."
        )

    return f"""
You are a friendly crop advisory chatbot assistant for Pakistani farmers.
You are having a general conversation — not answering a disease question.

LANGUAGE INSTRUCTION:
{language_instruction}

RULES:
1. Respond naturally and warmly in a single short paragraph — no bullet points or field labels.
2. Use the context from the retrieved document AND the conversation history to guide your response.
3. If it is a greeting — welcome the user and invite them to describe their crop problem.
4. If it is thanks — acknowledge warmly and offer continued help.
5. If it is farewell — wish them well and a good harvest.
6. If it is unclear — politely ask them to describe their crop, symptoms, and province.
7. If it is a weather or price question — politely say you cannot help with that
   and redirect them to describe a crop disease problem instead.
8. Keep response under 3 sentences. Do not use disease format labels.
"""


# ── NEW: system prompt for academic_general intent ────────────
def build_academic_general_prompt(language: str) -> str:
    lang = language.lower() if language else "english"

    if lang == "urdu":
        language_instruction = (
            "Always respond in Urdu script only. "
            "Translate all content into Urdu."
        )
    else:
        language_instruction = "Always respond in English only."

    return f"""
You are an expert agricultural advisor for Pakistani farmers.
The farmer has asked a GENERAL or ACADEMIC question — not a specific symptom.
Your job is to give a helpful OVERVIEW answer covering multiple aspects.

LANGUAGE INSTRUCTION:
{language_instruction}

RULES:
1. Give a comprehensive overview — list ALL relevant diseases/topics found in the retrieved documents.
2. For each disease, briefly mention: name, cause type (fungal/bacterial/viral), and key symptom.
3. Do NOT pick just one disease — cover ALL of them from the documents.
4. Format clearly with disease names as headings or a numbered/bulleted list.
5. End with a tip: "For specific treatment advice, describe the symptoms you are seeing."
6. Keep language simple and practical for a farmer.
7. If no documents found, answer from your general agricultural knowledge about Pakistan.
"""


def build_general_advisory_prompt(language: str) -> str:
    lang = language.lower() if language else "english"

    if lang == "urdu":
        language_instruction = (
            "Always respond in Urdu script only. "
            "Translate all content including numbers and dates into Urdu."
        )
    else:
        language_instruction = "Always respond in English only."

    return f"""
You are an expert agricultural advisor for Pakistani farmers.
Answer the farmer's question using ONLY the information in the retrieved document.

LANGUAGE INSTRUCTION:
{language_instruction}

RULES:
1. Answer clearly and directly using information from the document.
2. Include specific dates, doses, quantities, and schedules from the document.
3. Format as a short helpful paragraph — no disease format labels needed.
4. If Urdu, write the full response in Urdu script including all numbers and details.
5. Keep response concise and practical for a farmer.
"""


def generate_response(
    query: str,
    chunks: list,
    language: str = "english",
    confidence: str = "Low",
    conversation_history: list = None,
    intent: str = None          # NEW: passed from main.py so routing is always correct
) -> str:

    try:
        lang = language.lower() if language else "english"
        labels = LABELS.get(lang, LABELS["english"])

        # ── No chunks found at all ────────────────────────────
        if not chunks and intent != "academic_general":
            return labels["insufficient"]

        # ── FIX 1: Soft fallback for very-low confidence ──────
        top_score = chunks[0].get("similarity_score", 1.0) if chunks else 0
        is_very_low = top_score < 0.30

        # Intent from main.py takes priority; fall back to DB chunk_type
        top_chunk_type = intent if intent in ("academic_general", "price_market") \
                         else (chunks[0].get("chunk_type", "identification") if chunks else "identification")

        # ── Route: academic_general ───────────────────────────
        if top_chunk_type == "academic_general":
            print(f"🎓 Routing to academic_general handler")
            # Build context listing all retrieved disease names + summaries
            if chunks:
                overview_lines = []
                seen_diseases = set()
                for c in chunks:
                    dname = c.get("disease_name", "Unknown")
                    if dname in seen_diseases:
                        continue
                    seen_diseases.add(dname)
                    cause = c.get("cause", "N/A")
                    symptoms = c.get("symptoms", "N/A")
                    overview_lines.append(
                        f"- {dname}: Cause: {cause} | Symptoms: {symptoms}"
                    )
                context = "\n".join(overview_lines)
            else:
                context = "No specific documents found. Use your general agricultural knowledge about Pakistan."

            system_prompt = build_academic_general_prompt(language)
            user_message = f"""
Retrieved diseases from knowledge base:
{context}

Farmer's Question:
{query}

Give a comprehensive overview covering ALL diseases listed above.
"""

        # ── Route: small_talk ─────────────────────────────────
        elif top_chunk_type == "small_talk":
            print(f"💬 Routing to small_talk handler")
            context = chunks[0].get("chunk_text", "")
            system_prompt = build_smalltalk_prompt(language)
            user_message = f"""
Context from knowledge base:
{context}

User message:
{query}

Respond naturally and warmly in a single short paragraph.
"""

        # ── Route: general_advisory ───────────────────────────
        elif top_chunk_type == "general_advisory":
            print(f"🌾 Routing to general_advisory handler")
            context = chunks[0].get("chunk_text", "")
            system_prompt = build_general_advisory_prompt(language)
            user_message = f"""
Retrieved Advisory Document:
{context}

Farmer's Question:
{query}

Answer clearly using only the information above.
"""

        # ── Route: price_market ───────────────────────────────
        elif top_chunk_type == "price_market":
            print(f"💰 Routing to price_market handler")
            lang = language.lower() if language else "english"
            if lang == "urdu":
                user_message = f"کسان نے پوچھا: {query}"
                system_prompt = "آپ ایک فصل مشاورتی چیٹ بوٹ ہیں۔ قیمت یا منڈی کے سوالات کا جواب دینے سے معذرت کریں اور انہیں فصل کی بیماریوں کے بارے میں پوچھنے کی ہدایت کریں۔ اردو میں جواب دیں۔"
            else:
                user_message = f"Farmer asked: {query}"
                system_prompt = "You are a crop advisory chatbot. Politely explain you cannot provide market prices and redirect the farmer to ask about crop diseases or cultivation advice instead."

        # ── Route: disease (identification / management) ──────
        else:
            print(f"🔬 Routing to disease handler")
            context = build_context(chunks)
            system_prompt = build_system_prompt(language, confidence)
            # FIX 1: soft fallback preamble for very-low-confidence results
            soft_note = (
                "NOTE: The similarity score is very low. "
                "I found a related topic that may help. "
                "Please answer as best you can from the documents and "
                "add a disclaimer that the farmer should confirm with a local officer.\n\n"
            ) if is_very_low else ""
            user_message = f"""
Retrieved Agricultural Documents:
{context}

Farmer's Question:
{query}

{soft_note}IMPORTANT: Fill in ALL fields completely.
Do NOT leave Treatment or IPM Tips empty.
Copy the exact treatment and IPM information from the retrieved documents.
"""

        print(f"🤖 Sending query to Groq Llama 3.3 70B...")

        # Build messages list: system + prior history + current user message
        messages_payload = [{"role": "system", "content": system_prompt}]

        # Append prior conversation turns so the LLM has context
        if conversation_history:
            messages_payload.extend(conversation_history)

        # Append the current user message (with retrieved context)
        messages_payload.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_payload,
            temperature=0.1,
            max_tokens=800
        )

        response_text = response.choices[0].message.content
        print(f"✅ Groq response received successfully")
        return response_text

    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return "Sorry, the response generation service is temporarily unavailable. Please try again."


def test_llm():
    # Test 1: Disease query English
    sample_disease_chunks = [{
        "disease_name"       : "Yellow Rust",
        "crop"               : "wheat",
        "chunk_type"         : "identification",
        "cause"              : "Puccinia striiformis fungal pathogen",
        "symptoms"           : "Yellow-orange pustules in stripes on leaves",
        "chemical_control"   : "Propiconazole 25 EC at 0.5 liter per acre",
        "ipm_tips"           : "Use resistant varieties, monitor from January",
        "application_timing" : "At first pustule appearance on flag leaves",
        "resistant_varieties": ["NARC-2011", "Punjab-2011"],
        "yield_loss"         : "Up to 70 percent in susceptible varieties",
        "source"             : ["PARC NARC 2022", "Punjab Agriculture Extension 2023"],
        "similarity_score"   : 0.91
    }]

    # Test 2: Small talk chunk
    sample_smalltalk_chunks = [{
        "chunk_type"   : "small_talk",
        "disease_name" : "Greeting",
        "chunk_text"   : "User says hello salam. Bot is crop advisory assistant for Pakistani farmers.",
        "similarity_score": 0.95
    }]

    # Test 3: General advisory chunk
    sample_advisory_chunks = [{
        "chunk_type"   : "general_advisory",
        "disease_name" : "Wheat Sowing Advisory",
        "chunk_text"   : "Wheat sowing time Pakistan. Punjab KPK: October 15 to November 15. Sindh Balochistan: November 1 to November 30. Seed rate 50 kg per acre.",
        "similarity_score": 0.90
    }]

    print("=" * 50)
    print("TEST 1: Disease query English")
    print("=" * 50)
    print(generate_response("My wheat has yellow stripes", sample_disease_chunks, "english", "High"))

    print("\n" + "=" * 50)
    print("TEST 2: Small talk greeting Urdu")
    print("=" * 50)
    print(generate_response("السلام علیکم", sample_smalltalk_chunks, "urdu", "High"))

    print("\n" + "=" * 50)
    print("TEST 3: General advisory English")
    print("=" * 50)
    print(generate_response("When should I sow wheat in Punjab?", sample_advisory_chunks, "english", "High"))


if __name__ == "__main__":
    test_llm()