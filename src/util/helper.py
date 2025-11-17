import re, unicodedata
import hashlib
import json
from pathlib import Path
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import torch
import nltk
import os
from functools import lru_cache
import language_tool_python

nltk.download("punkt", quiet=True)

# =====================================================
#  UTILITY FUNCTIONS
# =====================================================

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
RAG_URL=""

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

RELATIVE_MODEL_PATH = Path(__file__).parent.parent / "models" / "fine-tuned-t5"
MODEL_PATH = str(RELATIVE_MODEL_PATH.resolve())
summarizer = pipeline("summarization", model=MODEL_PATH, tokenizer=MODEL_PATH, framework="pt")

# Factuality + Relevance models
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nli_model_name = "cross-encoder/nli-deberta-v3-base"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, token=os.getenv("HF_TOKEN"))
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, token=os.getenv("HF_TOKEN"))

sim_model = SentenceTransformer("all-MiniLM-L6-v2")

# PHI_MODEL = "microsoft/phi-3-mini-4k-instruct"

# phi_tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL)
# phi_model = AutoModelForCausalLM.from_pretrained(
#     PHI_MODEL,
#     torch_dtype="auto",
#     device_map="auto"
# )

# =====================================================
#  CLEANING
# =====================================================

import re
import nltk
from nltk.tokenize import sent_tokenize

def clean_article_max_strict(text: str) -> str:
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove common photo-credit patterns
    text = re.sub(r'(via|WireImage|FilmMagic|Getty Images|Image Group|Shutterstock)[^.,;]*', '', text, flags=re.IGNORECASE)

    # Force sentence tokenization
    sentences = sent_tokenize(text)

    cleaned = []

    for s in sentences:
        s_strip = s.strip()

        # Remove all-caps or title-case blocks (caption-like)
        if re.match(r'^[A-Z\s]{8,}$', s_strip):
            continue

        # Remove lines starting with photo/date/location formats
        if re.match(r'^(HOLLYWOOD|LOS ANGELES|NEW YORK|CALIFORNIA|NOVEMBER|DECEMBER|JANUARY|PHOTO).{0,40}:', s_strip, flags=re.IGNORECASE):
            continue

        # Remove malformed OCR-like fragments with too many capitalized words
        if len(re.findall(r'\b[A-Z][a-z]*\b', s_strip)) > 6:
            continue

        # Sentence MUST contain a verb
        if not re.search(r'\b(is|was|were|are|be|has|have|had|said|says|told|made|won|plays|attended|received|announced)\b', s_strip, flags=re.IGNORECASE):
            continue

        # Sentence must contain at least one noun
        if not re.search(r'\b[A-Za-z]+', s_strip):
            continue

        # Remove extremely short / empty lines
        if len(s_strip.split()) < 6:
            continue

        cleaned.append(s_strip)

    return " ".join(cleaned)

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

def summarize_text(body: str) -> dict:
    """Short, accurate summary (1–2 sentences)."""

    if not summarizer:
        return ""

    body = clean_article_max_strict(body)
    if not body:
        return ""

    # --- Smaller chunks = shorter + more accurate ---
    sentences = sent_tokenize(body)
    chunks, current = [], []

    for s in sentences:
        if sum(len(x.split()) for x in current) + len(s.split()) < 180:
            current.append(s)
        else:
            chunks.append(" ".join(current).strip())
            current = [s]

    if current:
        chunks.append(" ".join(current).strip())

    # --- Summaries (tight + factual) ---
    partials = []
    for ch in chunks:
        prompt = (
            "Summarize accurately in 1–2 concise sentences. "
            "No extra details, no speculation: "
        )

        out = summarizer(
            prompt + ch,
            max_length=60,      # short output
            min_length=20,      # avoids over-cutting
            num_beams=6,        # accuracy boost
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True
        )[0]["summary_text"].strip()

        out = trim_to_sentence(out)
        partials.append(out)

    # Join, clean, finalise
    raw = " ".join(partials)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = trim_to_sentence(raw)

    if raw:
        raw = raw[0].upper() + raw[1:]

    return raw

tool = language_tool_python.LanguageTool('en-US')

def fast_refine(summary: str) -> str:
    if not summary.strip():
        return summary
    summary = summary.replace("\n", " ").strip()
    corrected = tool.correct(summary)
    return corrected

def trim_to_sentence(text: str) -> str:
    text = text.strip()
    last_punct = max(text.rfind("."), text.rfind("?"), text.rfind("!"))
    if last_punct != -1:
        return text[: last_punct + 1]
    return text

# =====================================================
#  SAFE EXIT HOOK — Auto-save cache
# =====================================================
import atexit
@atexit.register
def _save_caches_on_exit():
    save_cache("factuality", factual_cache)
    save_cache("similarity", similarity_cache)