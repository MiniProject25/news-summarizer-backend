from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.articles.model.articleModel import Article, ArticleReference
from datetime import datetime
import asyncio
import numpy as np

from util.helper import summarize_text

# loading embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

SIMILARITY_THRESHOLD = 0.80

def embed(text: str):
    return model.encode(text).tolist()

def cos_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def parse_pgvector(vec):
    if isinstance(vec, list):
        return vec
    if isinstance(vec, str):
        return list(map(float, vec.strip("[]").split(",")))
    return vec


async def fast_duplicate_check(
    db: Session,
    title: str,
    body: str,
    source_url: str,
    source_name: str,
    source_bias: str,
    published_date: datetime,
    category: str,
    image_url: str,
    send_to_rag_fn,):

    combined_text = f"{title} {body}"
    embedding = await asyncio.to_thread(embed, combined_text)

    query = """
        SELECT
            article_id, title, summary, embedding
        FROM articles
        ORDER BY embedding <=> (:vec)::vector
        LIMIT 1
    """

    result = db.execute(
        text(query),
        {"vec": embedding}
    ).fetchone()

    existing_article_id = None
    existing_embedding = None

    if result:
        existing_article_id = result[0]
        existing_embedding = parse_pgvector(result[3])  # ← FIX HERE

    # DUPLICATE
    if existing_embedding is not None:
        similarity = cos_similarity(existing_embedding, embedding)
        if similarity >= SIMILARITY_THRESHOLD:

            existing_article = (
                db.query(Article)
                .filter(Article.article_id == existing_article_id)
                .first()
            )

            update_bias_count(existing_article, source_bias)

            db.add(ArticleReference(
                article_id=existing_article.article_id,
                source_url=source_url,
                published_date=published_date,
                source_name=source_name,
                bias=source_bias,
            ))
            db.commit()

            return True, existing_article

    # NEW ARTICLE
    new_article = Article(
        title=title,
        summary=await asyncio.to_thread(summarize_text, body),
        category=category,
        image_url=image_url,
        left_bias=1 if source_bias=="left" else 0,
        center_bias=1 if source_bias=="center" else 0,
        right_bias=1 if source_bias=="right" else 0,
        embedding=embedding,
    )

    db.add(new_article)
    db.commit()

    db.add(ArticleReference(
        article_id=new_article.article_id,
        source_url=source_url,
        published_date=published_date,
        source_name=source_name,
        bias=source_bias,
    ))
    db.commit()

    asyncio.create_task(
        send_to_rag_fn(
            new_article.article_id,
            body,
            title,
            source_url,
            new_article.category,
            str(published_date),
        )
    )

    return False, new_article


def update_bias_count(article: Article, bias: str):
    if bias == "left": 
        article.left_bias += 1 
    elif bias == "center": 
        article.center_bias += 1 
    elif bias == "right": 
        article.right_bias += 1