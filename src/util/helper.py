import hashlib
from pathlib import Path
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
import re
nltk.download("punkt")

## to generate content-hash - implementing deduplication
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

RELATIVE_MODEL_PATH = Path(__file__).parent.parent / "models" / "fine-tuned-bart"
MODEL_PATH = str(RELATIVE_MODEL_PATH.resolve())

summarizer = pipeline("summarization", model=MODEL_PATH, tokenizer=MODEL_PATH, framework="pt")

## remove unnecessary article meta-data
import re

def clean_article(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # --- Remove invisible and control characters ---
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # --- Remove typical newswire datelines, e.g., "TOKYO, JAPAN - SEPTEMBER 30:" ---
    text = re.sub(
        r"^[A-Z\s,]+-\s*(January|February|March|April|May|June|July|August|September|October|November|December)?"
        r"\s*\d{0,2}[:,]?\s*",
        "",
        text.strip()
    )

    # --- Remove “By AUTHOR” or “By John Doe, Reuters” ---
    text = re.sub(r"^By\s+[A-Z][A-Za-z\s\.,'-]{1,80}\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # --- Remove parenthetical photo or agency credits ---
    text = re.sub(r"\((Photo by|AP Photo|Getty Images|File|Reuters|Associated Press).*?\)", "", text, flags=re.IGNORECASE)

    # --- Remove email addresses ---
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "", text)

    # --- Remove publication timestamps ---
    text = re.sub(r"Published\s*[:\-–]\s*[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}.*?(AM|PM)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Updated\s*[:\-–].*?\d{4}.*?(AM|PM)?", "", text, flags=re.IGNORECASE)

    # --- Remove copyright and redistribution disclaimers ---
    text = re.sub(r"Copyright\s*©?\s*\d{4}.*?(All rights reserved\.?)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"This material may not be published.*?(redistributed\.)?", "", text, flags=re.IGNORECASE)

    # --- Remove “MORE FOR YOU”, “Related:”, “Recommended” sections ---
    text = re.sub(r"(MORE FOR YOU|Related|Recommended|Read more):.*", "", text, flags=re.IGNORECASE)

    # --- Remove common editorial suffixes like “___ contributed reporting” ---
    text = re.sub(r"[A-Z][A-Za-z\s,]+contributed to this report\.", "", text, flags=re.IGNORECASE)

    # --- Remove long runs of whitespace and normalize punctuation spacing ---
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([?.!,;:])", r"\1", text)

    return text.strip()

## summarizing article body
def summarize_text(body: str, max_chunk_tokens=512) -> str:
    # 1. Break the entire article into sentences 
    sentences = sent_tokenize(body)
    
    chunks = []
    current_chunk = ""

    # 2. Group sentences into chunks
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_tokens * 4:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk: # Add the last chunk
        chunks.append(current_chunk.strip())

    if not chunks:
        return ""

    # 3. Summarize each coherent chunk
    partial_summaries = []
    for chunk in chunks:
        if not chunk: continue
        summary = summarizer(chunk, max_new_tokens=200, min_length=40, do_sample=False)[0]["summary_text"]
        partial_summaries.append(summary)
    
    # 4. Combine and perform the final summarization
    combined = " ".join(partial_summaries)
    
    if not combined:
        return ""

    final = summarizer(combined, max_new_tokens=150, min_length=50, do_sample=False)[0]["summary_text"]
    
    # 5. Format the final output
    final_sentences = sent_tokenize(final)
    five_points = final_sentences[:5]
    return "\n".join([f"- {s.strip()}" for s in five_points])