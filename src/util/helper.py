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
def clean_article(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove datelines (e.g., "TOKYO, JAPAN - SEPTEMBER 30:")
    text = re.sub(r"^[A-Z\s,]+-\s*\w+\s*\d{1,2}:", "", text.strip())

    text = re.sub(r"\((Photo by|AP Photo).*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Copyright \d{4}.*?All rights reserved\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"This material may not be published.*?(redistributed\.)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(MORE FOR YOU:|Related:).*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)

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
        summary = summarizer(chunk, max_new_tokens=150, min_length=40, do_sample=False)[0]["summary_text"]
        partial_summaries.append(summary)
    
    # 4. Combine and perform the final summarization
    combined = " ".join(partial_summaries)
    
    if not combined:
        return ""

    final = summarizer(combined, max_new_tokens=120, min_length=50, do_sample=False)[0]["summary_text"]
    
    # 5. Format the final output
    final_sentences = sent_tokenize(final)
    five_points = final_sentences[:5]
    return "\n".join([f"- {s.strip()}" for s in five_points])