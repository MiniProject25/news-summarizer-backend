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
    # --- 1. Protect decimals like 7.8 or 2.4 ---
    text = re.sub(r"(\d)\.(\d)", r"\1DECIMAL\2", text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove photo-credits / agency fluff
    text = re.sub(
        r'(via|WireImage|FilmMagic|Getty Images|Image Group|Shutterstock)[^.,;]*',
        '',
        text,
        flags=re.IGNORECASE
    )

    sentences = sent_tokenize(text)
    cleaned = []

    for s in sentences:
        s_strip = s.strip()

        # Remove all-caps captions
        if re.match(r'^[A-Z\s]{8,}$', s_strip):
            continue

        # Remove location/photo metadata
        if re.match(
            r'^(HOLLYWOOD|LOS ANGELES|NEW YORK|CALIFORNIA|NOVEMBER|DECEMBER|JANUARY|PHOTO).{0,40}:',
            s_strip,
            flags=re.IGNORECASE
        ):
            continue

        # --- 2. Keep business sentences even if they lack a verb, as long as numbers appear ---
        if not re.search(r'\d', s_strip):
            # If no numbers, require verbs
            if not re.search(
                r'\b(is|was|were|are|be|has|have|had|said|says|told|made|won|plays|attended|received|announced|estimated|expected|reported|grew|grows|rises|rose|falls|fell)\b',
                s_strip,
                flags=re.IGNORECASE
            ):
                continue

        # At least 5 words
        if len(s_strip.split()) < 5:
            continue

        cleaned.append(s_strip)

    # Restore decimals
    final_text = " ".join(cleaned)
    final_text = final_text.replace("DECIMAL", ".")

    return final_text

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
T5_PREFIX = "summarize: Produce a very short, factual, bias-free, 1–2 sentence news summary: "

def summarize_text(body: str) -> str:
    """Ultra-short, factual 1–2 sentence summary."""

    if not summarizer:
        return ""

    body = clean_article_max_strict(body)
    if not body:
        return ""

    # --- Smaller chunks = shorter + tighter summaries ---
    sentences = sent_tokenize(body)
    chunks, current = [], []

    for s in sentences:
        if sum(len(x.split()) for x in current) + len(s.split()) < 80:
            current.append(s)
        else:
            chunks.append(" ".join(current).strip())
            current = [s]
    if current:
        chunks.append(" ".join(current).strip())

    partials = []
    for ch in chunks:
        prompt = (
            "summarize: Produce an extremely concise factual summary. "
            "Only capture the core point. "
            "Limit to one short sentence if possible. No elaboration: "
        )

        out = summarizer(
            prompt + ch,
            max_length=60,
            min_length=15,
            num_beams=4,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True
        )[0]["summary_text"].strip()

        out = trim_to_sentence(out)
        partials.append(out)

    # Join + tighten to max 2 sentences
    raw = " ".join(partials)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = trim_to_sentence(raw)

    # If summary somehow becomes long → compress once more
    if len(raw.split()) > 35:  # ~2 sentences limit
        raw = summarizer(
            "summarize: " + raw,
            max_length=40,
            min_length=15,
            num_beams=4,
            no_repeat_ngram_size=3,
            do_sample=False
        )[0]["summary_text"].strip()
        raw = trim_to_sentence(raw)

    # Capitalize
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