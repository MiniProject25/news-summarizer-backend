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
from util.deduplication_check import fast_duplicate_check
from util.helper import clean_article_max_strict, compute_hash, summarize_text
from util.articleHelper import categorize, send_to_rag_service
import asyncio
import requests
from bs4 import BeautifulSoup

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
    limit: int = 4, # for the purpose of smoke test
    db: Session = Depends(get_db),
):
    feeds = {
        # "Bloomberg" : { "url" : "https://www.bloomberg.com/feed/podcast/etf-report.xml", "bias" : "left" },
        # "Forbes" : {"url" : "https://www.forbes.com/business/feed/", "bias" : "center"},
        "CNBC" : {"url" : "https://www.cnbc.com/id/100003114/device/rss/rss.html", "bias" : "left"},
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

            count += 1
            
            try:
                url = entry.link
                # fetch article body
                news = NewsArticle(url, config=config)

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

                try:
                    headers = {
                        "User-Agent": config.browser_user_agent
                    }
                    raw_html = requests.get(url, headers=headers, timeout=10).text
                except Exception as e:
                    print(f"Failed HTML fetch {url} ({e})")
                    continue

                clean_body = None

                try:
                    import trafilatura
                    clean_body = trafilatura.extract(
                        raw_html,
                        include_comments=False,
                        favor_recall=False
                    )
                except:
                    clean_body = None

                if not clean_body:
                    try:
                        from readability import Document
                        clean_body = BeautifulSoup(
                            Document(raw_html).summary(), 
                            "lxml"
                        ).get_text("\n")
                    except:
                        clean_body = None

                if not clean_body:
                    try:
                        from boilerpy3 import extractors
                        extractor = extractors.ArticleExtractor()
                        clean_body = extractor.get_content(raw_html)
                    except:
                        clean_body = None

                if not clean_body:
                    clean_body = news.text

                if not clean_body or len(clean_body.strip()) < 100:
                    print(f"Body extraction failed or too short → {url}")
                    continue

                # print("Body before Cleaning: ")
                # print(body)
                
                # clean article body
                clean_body = clean_article_max_strict(clean_body)

                # print("Body after cleaning: ")
                # print(body)

                category = await categorize(clean_body)

                is_duplicate, article = await fast_duplicate_check(
                    db=db,
                    title=entry.title,
                    body=clean_body,
                    source_url=url,
                    source_bias=info["bias"],
                    source_name=source,
                    published_date=article_pubdate,
                    category= category,
                    image_url=news.top_image,
                    send_to_rag_fn=send_to_rag_service,
                )

            except Exception as e:
                print(f"Error scraping {entry.link}: {e}")
                continue
    
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
        
        return {
            "article_id": article.article_id,
            "title": article.title,
            "summary": article.summary,
            "category": article.category,
            "image_url": article.image_url,
            "left_bias": article.left_bias,
            "center_bias": article.center_bias,
            "right_bias": article.right_bias,
            "created_at": article.created_at.isoformat() if article.created_at else None,
            "views": article.views,
            "sources": article.sources,
        }

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

## ROUTE to fetch source breakdown for an article of id
@router.get("/source-breakdown/{article_id}")
async def get_breakdown(article_id: int, db: Session = Depends(get_db)):
    article = db.query(Article).filter(Article.article_id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    sources = (
        db.query(ArticleReference)
        .filter(ArticleReference.article_id == article_id)
        .all()
    )

    left = []
    center = []
    right = []

    ## sort them into the arrays
    for src in sources:
        item = {
            "name": src.source_name,
            "url": src.source_url,
            "publication_date": (
                src.published_date.isoformat() if src.published_date else ""
            )
        }

        if src.bias == "left":
            left.append(item)
        elif src.bias == "center":
            center.append(item)
        elif src.bias == "right":
            right.append(item)

    return {
        "left": left,
        "center": center,
        "right": right
    }

## ROUTE to fetch articles by category
@router.get("by-category")
async def get_articles_by_category(category: str, db: Session = Depends(get_db)):
    return(
        db.query(Article)
        .filter(Article.category.ilike(category))
        .order_by(Article.created_at.desc())
        .all()
    )