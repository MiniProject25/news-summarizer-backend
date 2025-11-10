import logging
from typing import Any, Dict, List
import feedparser
import httpx
from newspaper import Article as NewsArticle, Config
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from transformers import pipeline
from api.articles.model.articleModel import Article, ArticleReference
from config.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import or_
from schemas.schemas import ArticleSchema
import datetime
import time
import os
from util.helper import clean_article, compute_hash, summarize_text
from util.articleHelper import send_to_rag_service
import asyncio

router = APIRouter() 

## get articles based on search
@router.get("/articles/")
def getAllArticles(
    search: str = Query(None, description="Search Article"), 
    db: Session = Depends(get_db)
    # Query() helps in defining additional constraints (validation, default value setup, min-length, max-length)
):
    # store final list of articles
    result = []

    # get all the articles from the database
    query = db.query(Article)

    # filter out to match the search prefix (if provided)
    if search:
        search_term = f"%{search.lower()}%"
        query = query.filter(
            or_(
                Article.title.ilike(search_term),
                Article.summary.ilike(search_term)
            )
        )

    # get the latest articles first
    articles = query.order_by(Article.created_at.desc()).all()

    # format according to database schema
    for article in articles:
        result.append({
            "article_id" : article.article_id,
            "title" : article.title,
            "summary" : article.summary,
            "category" : article.category,
            "image_url" : article.image_url,
            "left_bias" : article.left_bias,
            "right_bias" : article.right_bias,
            "center_bias" : article.center_bias,
            "created_at" : article.created_at,
            "views" : article.views,
            "sources" : article.sources,
        })

    return result
## scrape articles from multiple rss feeds
## to be run as a periodic job

## rss feed list source: https://themeisle.com/blog/rss-feeds-list/#business

## bias source: [
#   https://www.allsides.com/media-bias/media-bias-chart,
#   https://adfontesmedia.com/  
#  ]

@router.get("/scrape")
async def scrapeArticles(
    limit: int = 2, # for the purpose of smoke test
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    feeds = {
        # "Bloomberg" : { "url" : "https://www.bloomberg.com/feed/podcast/etf-report.xml", "bias" : "left" },
        "Forbes" : {"url" : "https://www.forbes.com/business/feed/", "bias" : "center"},
        # "CNBC" : {"url" : "https://www.cnbc.com/id/100003114/device/rss/rss.html", "bias" : "left"},
        # "Financial Times" : {"url" : "https://www.ft.com/?format=rss", "bias" : "center"}, 
        # "The Economist" : {"url" : "https://www.economist.com/latest/rss.xml", "bias" : "left"}, 
        # "Harvard Business Reviews" : {"url" : "https://hbr.org/feed", "bias" : "center"}, 
        # "Market Watch" : {"url" : "https://www.marketwatch.com/rss/topstories", "bias" : "center"}, 
        # "Wall Street Journal - Business" : {"url" : "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "bias" : "center"}, 
        # "Business Insider" : {"url" : "https://www.businessinsider.com/rss", "bias" : "center"}, 
        # "Investopedia" : {"url" : "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_articles", "bias" : "center"} 
    }

    config = Config()
    config.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
    )
    config.request_timeout = 10

    count = 0  # smoke test
    ## fetch articles from the rss feeds of various sources
    for source, info in feeds.items():
        feed = feedparser.parse(info["url"])

        for entry in feed.entries:

            if count >= limit:
                return {"message" : f"Scrape completed for {count} articles"}

            try:
                # fetch article body
                news = NewsArticle(entry.link, config=config)

                article_pubdate = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    date_tuple = entry.published_parsed
                    article_pubdate = datetime.datetime.fromtimestamp(time.mktime(date_tuple))

                try:
                    news.download()
                    news.parse()
                except Exception as e:
                    print(f"Skipping {entry.link}: parse failed ({e})")
                    continue

                body = await asyncio.to_thread(lambda: news.text.strip())

                if not body:
                    print(f"Empty body for {entry.link}")
                    continue

                print("Body before Cleaning: ")
                print(body)
                
                # clean article body
                body = clean_article(news.text)

                print("Body after cleaning: ")
                print(body)

                # deduplication check
                content_hash = compute_hash(body)
                existing_article = db.query(Article).filter(Article.content_hash == content_hash).first()

                if existing_article:
                    # update bias_count
                    if info["bias"] == "left":
                        existing_article.left_bias += 1
                    if info["bias"] == "center":
                        existing_article.center_bias += 1
                    if info["bias"] == "right":
                        existing_article.right_bias += 1

                    # add a new reference
                    new_ref = ArticleReference(
                        article_id=existing_article.article_id,
                        source_url=entry.link,
                        published_date=article_pubdate,
                        source_name=source,
                        bias=info["bias"]
                    )

                    db.add(new_ref)
                else:                    
                    # summarize the body
                    summary = await asyncio.to_thread(summarize_text, body)

                    if not summary.strip() or len(summary.split()) < 5:
                        summary = "No summary available for this article."

                    new_article = Article(
                        title=entry.title,
                        summary=summary,
                        category="finance", # for now
                        image_url=news.top_image,
                        left_bias=1 if info["bias"] == "left" else 0,
                        center_bias=1 if info["bias"] == "center" else 0,
                        right_bias=1 if info["bias"] == "right" else 0,
                        content_hash=content_hash
                    )

                    db.add(new_article)
                    db.commit()
                    db.refresh(new_article)

                    article_pubdate = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        date_tuple = entry.published_parsed
                        article_pubdate = datetime.datetime.fromtimestamp(time.mktime(date_tuple))

                    # send to RAG service for generating embeddings
                    background_tasks.add_task(
                        send_to_rag_service,
                        new_article.article_id,
                        body,
                        entry.title,
                        entry.link,
                        new_article.category,
                        str(article_pubdate)
                    )

                    new_ref = ArticleReference(
                        article_id=new_article.article_id,
                        source_url=entry.link,
                        published_date=article_pubdate,
                        source_name=source,
                        bias=info["bias"]
                    )

                    db.add(new_ref)

                db.commit()

            except Exception as e:
                print(f"Error scraping {entry.link}: {e}")
                continue

            count += 1
    
    return {"message" : "Scraping completed"}

## get list of article details
@router.get("/details")
def getArticlesByIds(
    ids: str = Query(..., description="Comma-separated list of article IDs"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    try:
        # Split on commas and convert to integers
        id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
        
        if not id_list:
            raise HTTPException(status_code=400, detail="No valid article IDs provided")
        
        articles = db.query(Article).filter(Article.article_id.in_(id_list)).all()

        if not articles:
            raise HTTPException(status_code=404, detail="Articles not found")
        
        related_articles_data = []
        for article in articles:
            related_articles_data.append({
                # Map DB fields to frontend fields
                "id": str(article.article_id), 
                "title": article.title,
                # Assuming your Article model has a field for the URL path
                "url": f"/Home/{article.article_id}" # <-- Use your app's internal URL structure
            })

        return related_articles_data
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format provided")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

## get individual article details
@router.get("/{id}")
def getArticleById(id: int, db: Session = Depends(get_db)):
    try:
        article = db.query(Article).filter(Article.article_id == id).first()

        if article is None:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return article
    except Exception as e:
        print(f"Error: {e}")



## Handling blocking operations 
# in the above route, 
# 1. feedparser.parse()
# 2. newspaper.download(), .parse() 
# 3. summarize_text()
# 4. db.query(), db.add(), db.commit() 
# all the above operations run synchronously
# This conflicts with the asynchronous nature of the above route

## Strategy to resolve this,
# make the route synchronous
# use asyncio library to use concurrency


## ROUTE to increment article view count
@router.post("/{article_id}/increment_view")
def increment_views(article_id: int, db: Session = Depends(get_db)):
    article = db.query(Article).filter(Article.article_id == article_id).first()
    if not article:
        db.close()
        raise HTTPException(status_code=404, detail="Article not found")

    article.views += 1
    db.commit()
    db.refresh(article)
    db.close()

    return {"article_id": article_id, "views": article.views}


## ROUTE to perform similarity search through RAG microservice
@router.get("/similar-articles/{article_id}")
async def get_similar_articles(article_id: int, n: int = 5):
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{os.getenv("RAG_URL")}/query", json={"id": str(article_id), "top_k": n})
            return r.json()
        except Exception as e:
            logging.exception("Failed to fetch similar articles")
            return {"error": str(e)}