import asyncio
import os
import httpx
import logging

from transformers import pipeline

RAG_ROUTE=os.getenv("RAG_URL")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

CATEGORIES = [
    "finance", "politics", "business", "tech",
    "world", "sports", "entertainment",
    "science", "health"
]

async def send_to_rag_service(article_id, content, title, url, category, date):
    logging.warning(f"Sending article {article_id} to RAG for embedding...")

    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{RAG_ROUTE}/embed", # LLM RAG service endpoint
                json={
                    "id": article_id,
                    "content": content,
                    "title": title,
                    "url": url,
                    "category": category,
                    "date": date
                }
            )
            logging.warning(f"RAG responded: {res.status_code}")
        except Exception as e:
            logging.exception(f"Failed to send embedding request: {e}")
            raise

async def categorize(text: str):
    result = await asyncio.to_thread(lambda: classifier(text, CATEGORIES))
    return result["labels"][0]