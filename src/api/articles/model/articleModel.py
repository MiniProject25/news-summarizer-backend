from sqlalchemy import Column, Integer, Text, String, TIMESTAMP, ForeignKey, Date
from sqlalchemy.orm import relationship
from datetime import datetime
from config.database import Base

class Article(Base):
    __tablename__ = "articles"
        
    article_id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    summary = Column(Text)
    category = Column(String(100))
    image_url = Column(Text)
    left_bias = Column(Integer)
    right_bias = Column(Integer)
    center_bias = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.utcnow())
    views = Column(Integer, default=0)
    content_hash = Column(Text, unique=True)

        # relationship with references
    sources = relationship("ArticleReference", back_populates="article", cascade="all, delete-orphan")

class ArticleReference(Base):
    __tablename__ = "article_references"
    reference_id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.article_id"), nullable=False)
    source_url = Column(Text)
    published_date = Column(Date)
    source_name = Column(String(150))
    bias = Column(String(20))

    article = relationship("Article", back_populates="sources")
