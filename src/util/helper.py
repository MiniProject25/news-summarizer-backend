import re, unicodedata
import hashlib
import json
from pathlib import Path
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import torch
import nltk
import os
from functools import lru_cache
nltk.download("punkt", quiet=True)

# =====================================================
#  UTILITY FUNCTIONS
# =====================================================

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"

def load_cache(name: str):
    path = cache_path(name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(name: str, data: dict):
    with open(cache_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =====================================================
#  MODEL INITIALIZATION
# =====================================================

RELATIVE_MODEL_PATH = Path(__file__).parent.parent / "models" / "fine-tuned-bart"
MODEL_PATH = str(RELATIVE_MODEL_PATH.resolve())
summarizer = pipeline("summarization", model=MODEL_PATH, tokenizer=MODEL_PATH, framework="pt")

# Factuality + Relevance models
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nli_model_name = "cross-encoder/nli-deberta-v3-base"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, token=os.getenv("HF_TOKEN"))
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, token=os.getenv("HF_TOKEN"))

sim_model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
#  CLEANING
# =====================================================

def clean_article(text: str) -> str:
    """Aggressive text cleanup: removes junk, metadata, boilerplate."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize Unicode characters first
    text = unicodedata.normalize("NFKC", text)
    
    # Remove invisible and special whitespace chars
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Replace common HTML entities
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", text)

    # Remove photo credits, agencies, etc.
    text = re.sub(r"(?i)\b(FILE PHOTO|NOTE TO USER|AP Photo|Getty Images|Reuters|AFP|Shutterstock|Courtesy of)\b.*", "", text)
    
    # Remove leading date/location headers
    text = re.sub(r"^\s*[-–—]*\s*[A-Z\s,.'’\-]+ - ", "", text)
    text = re.sub(r"^\s*(?:[A-Z]+\s+\d{1,2}:|[A-Z][a-z]+,\s*[A-Z][a-z]+\s*-\s*)", "", text)

    # Remove bylines
    text = re.sub(r"By\s+[A-Z][A-Za-z\s\.,'’\-]{1,80}", "", text)

    # Remove photo/video captions
    text = re.sub(r"\((Photo|Image|Video) by .*?\)", "", text, flags=re.IGNORECASE)

    # Remove URLs, tags, and mentions
    text = re.sub(r"https?://\S+|@[A-Za-z0-9_]+|#[A-Za-z0-9_]+", "", text)

    # Remove boilerplate/news site sections
    text = re.sub(
        r"(?i)\b(MORE FOR YOU|READ MORE|SEE ALSO|TOP STORIES|TRENDING|RECOMMENDED|WATCH NOW|ADVERTISEMENT|SPONSORED)\b.*",
        "", text)
    text = re.sub(r"(All rights reserved|This material may not be published|Advertisement).*", "", text, flags=re.IGNORECASE)

    # Normalize dashes, but keep quotes
    text = re.sub(r"–|—", "-", text)
    text = re.sub(r"[\[\]<>]", "", text)  # removed () from here to keep quotes safe

    # Fix spaces around punctuation
    text = re.sub(r"\s([?.!,;:])", r"\1", text)

    # Remove non-ASCII replacement chars but preserve smart quotes
    text = text.encode("utf-8", "replace").decode("utf-8")
    text = re.sub(r"[�\uFFFD]+", "", text)

    # Collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Fix truncated endings like ". word"
    text = re.sub(r"\.\s*\w*$", ".", text)
    
    return text.strip()

# =====================================================
#  FACTUALITY + SEMANTIC CHECKS (CACHED)
# =====================================================

factual_cache = load_cache("factuality")
similarity_cache = load_cache("similarity")

@lru_cache(maxsize=2048)
def is_factually_supported(article: str, sentence: str, threshold: float = 0.7) -> bool:
    """Checks entailment using DeBERTa; caches results."""
    key = compute_hash(article[:3000] + sentence)
    if key in factual_cache:
        return factual_cache[key]

    inputs = nli_tokenizer(article, sentence, return_tensors="pt",
                           truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze()
    result = probs[2].item() >= threshold  # entailment label

    factual_cache[key] = result
    if len(factual_cache) % 20 == 0:
        save_cache("factuality", factual_cache)
    return result


@lru_cache(maxsize=2048)
def is_relevant(article: str, sentence: str, min_sim: float = 0.45) -> bool:
    """Semantic similarity filter; cached for efficiency."""
    key = compute_hash(article[:3000] + sentence)
    if key in similarity_cache:
        return similarity_cache[key]

    emb_a = sim_model.encode(article, convert_to_tensor=True)
    emb_b = sim_model.encode(sentence, convert_to_tensor=True)
    sim = util.cos_sim(emb_a, emb_b).item()
    result = sim >= min_sim

    similarity_cache[key] = result
    if len(similarity_cache) % 20 == 0:
        save_cache("similarity", similarity_cache)
    return result

# =====================================================
#  SUMMARIZATION
# =====================================================

def summarize_text(body: str, max_chunk_tokens=512) -> str:
    """Generates 3–6 factual, coherent, non-redundant bullet points."""
    body = clean_article(body)
    if not body:
        return ""

    # ---- Chunking ----
    sentences = sent_tokenize(body)
    chunks, current_chunk = [], ""
    for s in sentences:
        if len(current_chunk) + len(s) < max_chunk_tokens * 4:
            current_chunk += " " + s
        else:
            chunks.append(current_chunk.strip())
            current_chunk = s
    if current_chunk:
        chunks.append(current_chunk.strip())

    # ---- Summarize each chunk ----
    partials = []
    for ch in chunks:
        summary = summarizer(
            ch, max_new_tokens=90, min_length=30, do_sample=False, truncation=True
        )[0]["summary_text"]
        partials.append(summary.strip())

    # ---- Merge & clean ----
    combined = " ".join(partials)
    combined = re.sub(r"\s+", " ", combined).strip()

    # ---- Split into potential bullet points ----
    bullets = re.split(r'(?<=[.!?])\s+', combined)
    bullets = [b.strip() for b in bullets if 5 < len(b.split()) < 40]

    # ---- Deduplicate & filter junk ----
    seen, cleaned = set(), []
    for b in bullets:
        b_lower = b.lower()
        if any(x in b_lower for x in ["getty", "photo", "note to user", "copyright", "courtesy", "file photo"]):
            continue
        if b_lower in seen:
            continue
        seen.add(b_lower)
        b = re.sub(r"\b([A-Za-z]+) \1\b", r"\1", b)
        b = re.sub(r"\s+", " ", b).strip(" -")
        cleaned.append(b)

    # ---- Quality filters ----
    final_bullets = []
    for b in cleaned[:10]:
        if is_factually_supported(body, b) and is_relevant(body, b):
            final_bullets.append(b)

    final_bullets = final_bullets[:6]
    if not final_bullets:
        return ""
    return "\n- " + "\n- ".join(final_bullets)

# =====================================================
#  SAFE EXIT HOOK — Auto-save cache
# =====================================================
import atexit
@atexit.register
def _save_caches_on_exit():
    save_cache("factuality", factual_cache)
    save_cache("similarity", similarity_cache)