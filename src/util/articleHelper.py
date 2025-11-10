import os
import httpx
import logging

RAG_ROUTE=os.getenv("RAG_URL")

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
