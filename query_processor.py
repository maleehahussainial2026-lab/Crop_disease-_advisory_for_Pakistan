# ============================================================
# query_processor.py
# Updated: Added intent classification to handle academic/general
# questions differently from farmer symptom queries.
#
# Intent types:
#   "farmer_symptom"   → RAG retrieval (disease, treatment)
#   "general_advisory" → RAG retrieval (cultivation, fertilizer, season)
#   "academic_general" → LLM answers directly (no RAG needed)
#   "small_talk"       → RAG retrieval (greeting, farewell, thanks)
#   "weather_season"   → RAG retrieval (new weather dataset)
#   "price_market"     → LLM politely declines, redirects
#
# Usage:
#   from query_processor import preprocess_query, classify_intent
#   intent = classify_intent(query)
#   clean_query = preprocess_query(query)
# ============================================================

import re

# ── Try to import pyspellchecker (optional dependency) ───────
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    SPELL_AVAILABLE = True
except ImportError:
    SPELL_AVAILABLE = False
    print("⚠️  pyspellchecker not installed — spell correction disabled")
    print("   Install with: pip install pyspellchecker")


# ============================================================
# SECTION 1: INTENT CLASSIFICATION
# ============================================================

# ── Keywords that signal an ACADEMIC / GENERAL question ──────
# These are questions a teacher, student, or researcher would ask.
# The LLM can answer these directly without RAG retrieval.
ACADEMIC_KEYWORDS = {
    # Conceptual / definitional questions
    "what is", "what are", "define", "definition", "explain",
    "difference between", "how does", "how do", "tell me about",
    "describe", "meaning of", "types of", "examples of",
    "introduction to", "overview of", "summary of",

    # Academic crop/disease knowledge
    "ipm", "integrated pest management",
    "fungicide", "pesticide", "bactericide", "herbicide",
    "seed treatment", "crop rotation", "resistant varieties",
    "rabi", "kharif", "rabi season", "kharif season",
    "economic importance", "significance of",
    "which province", "which provinces",
    "major crops", "main crops", "crops of pakistan",
    "how does disease spread", "how does fungal",
    "life cycle of", "pathogen", "pathology",
    "parc", "aari", "agriculture department",
    "food security", "yield loss", "production",
    "agronomy", "horticulture", "botany",

    # Named diseases asked academically (not symptom-based)
    "what is yellow rust", "what is brown rust",
    "what is karnal bunt", "what is rice blast",
    "what is loose smut", "what is powdery mildew",
    "what is leaf curl", "what is sheath blight",
    "what is bacterial blight", "what is late blight",
    "what is early blight", "what is fusarium wilt",
    "what is red rot", "what is sugarcane smut",
    "what is ratoon stunting", "what is fall armyworm",
    "what is stem borer", "what is whitefly",
    "what is aphid", "what is thrips", "what is pyrilla",
    "tell me about disease", "names of diseases",
    "list of diseases", "diseases of wheat", "diseases of rice",
    "diseases of cotton", "diseases of maize",
    "diseases of sugarcane", "diseases of potato",
    "diseases of tomato", "diseases of onion",
}

# ── Keywords that signal a FARMER SYMPTOM query ──────────────
# These need RAG retrieval for precise, evidence-based answers.
SYMPTOM_KEYWORDS = {
    # Symptom descriptions
    "spots", "lesions", "yellowing", "wilting", "dying",
    "rotting", "powder", "pustules", "stripes", "blight",
    "curling", "stunted", "lodging", "discoloration",
    # Roman Urdu symptoms
    "daag", "dhabbe", "peela", "sukha", "murjha", "jala",
    "kharab", "bimari", "masla", "nuksan",
    # Treatment seeking
    "how to treat", "how to control", "what spray", "what medicine",
    "which fungicide", "which pesticide", "treatment for",
    "spray for", "dawa", "ilaj", "spray karo", "kya karein",
    # Urgency markers
    "my crop", "my wheat", "my rice", "meri fasal",
    "mera khet", "meri gehun", "help me", "what to do",
    "emergency", "all plants dying", "spreading fast",
}

# ── Keywords that signal WEATHER / SEASON questions ──────────
WEATHER_KEYWORDS = {
    "weather", "mausam", "season", "monsoon", "rain", "barish",
    "temperature", "humidity", "frost", "heat wave", "garmi", "sardi",
    "kharif season", "rabi season", "crop calendar", "sowing time",
    "when to sow", "when to plant", "when to harvest",
    "harvest time", "planting time", "best time",
    "which month", "which season", "fasal ka waqt",
    "fasl kab lagaein", "kab spray karein",
    "disease in monsoon", "disease in winter", "disease in summer",
    "barish ke baad", "garmi mein bimari", "sardi mein bimari",
}

# ── Keywords that signal PRICE / MARKET questions ─────────────
# Bot cannot answer these — politely redirect.
PRICE_KEYWORDS = {
    "price", "rate", "mandi", "market price", "cost", "bhav",
    "khareed", "sell", "bech", "income", "profit", "loss amount",
    "rupees", "rs.", "per kg", "per maund", "subsidy",
    "government rate", "support price", "wheat price", "cotton price",
}

# ── Keywords that signal SMALLTALK ───────────────────────────
SMALLTALK_KEYWORDS = {
    "hello", "hi", "salam", "assalam", "adaab", "sat sri",
    "thank", "shukriya", "mehrbani", "shukria",
    "goodbye", "bye", "khuda hafiz", "allah hafiz", "alvida",
    "who are you", "what can you do", "aap kaun hain",
    "are you a doctor", "are you real", "aap kya kar sakte",
    "you are great", "very helpful",
}


def classify_intent(query: str) -> str:
    """
    Classify the user query into one of these intent types:
      - "academic_general"  → LLM answers directly (no RAG)
      - "farmer_symptom"    → RAG disease retrieval
      - "general_advisory"  → RAG general advisory retrieval
      - "weather_season"    → RAG weather dataset retrieval
      - "price_market"      → LLM politely declines
      - "small_talk"        → RAG smalltalk retrieval

    Returns the intent string.
    """
    q = query.lower().strip()

    # ── Check price/market first (clear non-agricultural domain) ─
    for kw in PRICE_KEYWORDS:
        if kw in q:
            print(f"💰 Intent: price_market (matched '{kw}')")
            return "price_market"

    # ── Check smalltalk ──────────────────────────────────────────
    for kw in SMALLTALK_KEYWORDS:
        if kw in q:
            print(f"💬 Intent: small_talk (matched '{kw}')")
            return "small_talk"

    # ── Check weather/season ─────────────────────────────────────
    for kw in WEATHER_KEYWORDS:
        if kw in q:
            print(f"🌤️  Intent: weather_season (matched '{kw}')")
            return "weather_season"

    # ── Check academic/general ───────────────────────────────────
    for kw in ACADEMIC_KEYWORDS:
        if kw in q:
            print(f"🎓 Intent: academic_general (matched '{kw}')")
            return "academic_general"

    # ── Check farmer symptom ─────────────────────────────────────
    for kw in SYMPTOM_KEYWORDS:
        if kw in q:
            print(f"🌾 Intent: farmer_symptom (matched '{kw}')")
            return "farmer_symptom"

    # ── Default: treat as farmer symptom (safe fallback) ────────
    print(f"🔄 Intent: farmer_symptom (default fallback)")
    return "farmer_symptom"


# ============================================================
# SECTION 2: QUERY PREPROCESSING (original logic, unchanged)
# ============================================================

ROMAN_URDU_SYNONYMS = {
    # Crops
    "gehun"      : "wheat",
    "gandum"     : "wheat",
    "chawal"     : "rice",
    "dhan"       : "rice",
    "kapas"      : "cotton",
    "makkai"     : "maize",
    "makka"      : "maize",
    "corn"       : "maize",
    "alu"        : "potato",
    "aloo"       : "potato",
    "tamatar"    : "tomato",
    "piaz"       : "onion",
    "ganna"      : "sugarcane",

    # Plant parts
    "patta"      : "leaf",
    "pattay"     : "leaves",
    "patton"     : "leaves",
    "tana"       : "stem",
    "tanay"      : "stem",
    "jar"        : "root",
    "phool"      : "flower",
    "phal"       : "fruit",
    "bali"       : "grain ear spike",
    "daana"      : "grain seed",

    # Symptoms — colours
    "peela"      : "yellow yellowing",
    "peeli"      : "yellow yellowing",
    "peela rang" : "yellow color yellowing",
    "laal"       : "red",
    "kala"       : "black",
    "kali"       : "black",
    "safed"      : "white",
    "bhoora"     : "brown",
    "bhura"      : "brown",
    "narangi"    : "orange",

    # Symptom descriptions
    "daag"       : "spot lesion patch",
    "dhabbe"     : "spots lesions patches",
    "dhabba"     : "spot lesion",
    "dag"        : "spot lesion",
    "jala"       : "blight burn",
    "murjhana"   : "wilting wilt",
    "murjha"     : "wilting wilt",
    "sukh"       : "drying dry wilting",
    "sukha"      : "drought dry wilting",
    "sada"       : "rot rotten",
    "sarna"      : "rot decay",
    "sara"       : "rot rotten",
    "galna"      : "rot decay",
    "jhurriyan"  : "wrinkles shriveling",
    "chhote"     : "small",
    "bade"       : "large",
    "jhar"       : "falling dropping",
    "gir"        : "falling dropping",
    "kata"       : "cut lesion",
    "powder"     : "powdery mildew",
    "powdery"    : "powdery mildew",
    "phaptund"   : "fungus mold",
    "fungi"      : "fungal fungus",
    "keeray"     : "insects pests",
    "keera"      : "insect pest",
    "machar"     : "aphid insect",

    # Conditions / timing
    "barish"     : "rain rainfall",
    "nami"       : "humidity moisture",
    "garmi"      : "heat temperature warm",
    "sardi"      : "cold winter",
    "mausam"     : "weather season",
    "fasal"      : "crop plant",
    "khait"      : "field farm",
    "khet"       : "field farm",

    # Problem words
    "kharab"     : "damaged diseased problem",
    "bimari"     : "disease",
    "masla"      : "problem issue",
    "nuksan"     : "damage loss",
    "halat"      : "condition state",
    "ilaj"       : "treatment cure",
    "dawa"       : "medicine pesticide fungicide",
    "spray"      : "spray fungicide pesticide",
    "khad"       : "fertilizer",

    # Fillers / prepositions (strip these)
    "mein"       : "",
    "pe"         : "",
    "par"        : "",
    "ki"         : "",
    "ka"         : "",
    "ke"         : "",
    "hai"        : "",
    "hain"       : "",
    "ho"         : "",
    "raha"       : "",
    "rahi"       : "",
    "kuch"       : "",
    "aur"        : "",
    "nahi"       : "",
    "nahin"      : "",
    "bhi"        : "",
    "se"         : "",
    "ko"         : "",
    "ne"         : "",
    "tha"        : "",
    "thi"        : "",
    "wala"       : "",
    "wali"       : "",
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "so", "if", "my", "our",
    "your", "its", "this", "that", "these", "those", "there",
    "i", "we", "you", "he", "she", "they", "it", "me", "us",
    "what", "how", "why", "when", "where", "which", "who",
    "please", "tell", "me", "help", "about", "any", "some",
    "mein", "pe", "par", "ki", "ka", "ke", "hai", "hain", "ho",
    "raha", "rahi", "kuch", "aur", "nahi", "nahin", "bhi",
    "se", "ko", "ne", "tha", "thi", "wala", "wali", "ap",
    "aap", "mera", "meri", "mere", "ek", "do", "teen",
}


def _contains_urdu_script(text: str) -> bool:
    for ch in text:
        if '\u0600' <= ch <= '\u06FF':
            return True
    return False


def _tokenize(text: str) -> list:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = re.split(r'[\s,.\-?!؟،]+', text)
    return [t for t in tokens if t]


def _spell_correct_token(token: str) -> str:
    if not SPELL_AVAILABLE:
        return token
    if len(token) <= 2 or token in ROMAN_URDU_SYNONYMS:
        return token
    if any(c.isdigit() for c in token):
        return token
    corrected = _spell.correction(token)
    return corrected if corrected else token


def preprocess_query(query: str) -> str:
    """
    Enrich the query for embedding:
    1. Detect and keep Urdu script tokens
    2. Tokenize Roman Urdu / English
    3. Strip stopwords
    4. Spell-correct English tokens
    5. Append English synonyms for Roman Urdu tokens
    6. De-duplicate and rejoin
    """
    if not query or not query.strip():
        return query

    original = query.strip()
    print(f"🔤 Preprocessing query: '{original}'")

    urdu_part  = []
    roman_part = []

    for word in original.split():
        if _contains_urdu_script(word):
            urdu_part.append(word)
        else:
            roman_part.append(word)

    tokens   = _tokenize(" ".join(roman_part))
    synonyms = []
    cleaned_tokens = []

    for token in tokens:
        if token in STOPWORDS:
            continue
        if token in ROMAN_URDU_SYNONYMS:
            synonym = ROMAN_URDU_SYNONYMS[token]
            if synonym:
                synonyms.append(synonym)
            cleaned_tokens.append(token)
        else:
            corrected = _spell_correct_token(token)
            if corrected != token:
                print(f"   ✏️  Spell corrected: '{token}' → '{corrected}'")
            cleaned_tokens.append(corrected)

    parts = []
    if urdu_part:
        parts.append(" ".join(urdu_part))
    if cleaned_tokens:
        parts.append(" ".join(cleaned_tokens))
    if synonyms:
        parts.append(" ".join(synonyms))

    enriched = " ".join(parts)

    seen  = set()
    final = []
    for word in enriched.split():
        if word not in seen:
            seen.add(word)
            final.append(word)

    result = " ".join(final)
    print(f"   ✅ Enriched query: '{result}'")
    return result


# ============================================================
# SECTION 3: SELF-TEST
# ============================================================

if __name__ == "__main__":
    test_cases = [
        # Academic / general
        ("What is Yellow Rust?",              "academic_general"),
        ("Tell me about diseases of wheat",   "academic_general"),
        ("What is IPM?",                      "academic_general"),
        ("Diseases of rice in Pakistan",      "academic_general"),
        ("What is Karnal Bunt disease?",      "academic_general"),
        ("Which province grows cotton?",      "academic_general"),
        ("Difference between Rabi and Kharif","academic_general"),
        ("What are resistant varieties?",     "academic_general"),

        # Weather / season
        ("When should I sow wheat in Punjab?","weather_season"),
        ("Kharif season crops Pakistan",      "weather_season"),
        ("Which diseases occur in monsoon?",  "weather_season"),
        ("Mausam mein kaunsi bimari hoti hai","weather_season"),
        ("Best time to harvest rice",         "weather_season"),

        # Farmer symptom
        ("My wheat has yellow stripes on leaves", "farmer_symptom"),
        ("gehun k patton pe peela rang",          "farmer_symptom"),
        ("Rice plants are wilting and dying",     "farmer_symptom"),
        ("What spray for cotton leaf curl?",      "farmer_symptom"),
        ("Meri fasal kharab ho rahi hai",         "farmer_symptom"),

        # Price
        ("What is the price of wheat today?", "price_market"),
        ("Cotton mandi rate kya hai?",         "price_market"),

        # Smalltalk
        ("Hello, I need help",                "small_talk"),
        ("Shukriya for your help",            "small_talk"),
        ("Khuda Hafiz",                       "small_talk"),
    ]

    print("=" * 65)
    print("INTENT CLASSIFIER SELF-TEST")
    print("=" * 65)
    passed = 0
    failed = 0
    for query, expected in test_cases:
        result = classify_intent(query)
        status = "✅ PASS" if result == expected else f"❌ FAIL (got: {result})"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status} | Expected: {expected}")
        print(f"           Query: {query}")
        print()

    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")