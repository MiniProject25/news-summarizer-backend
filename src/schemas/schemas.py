from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# These schemas help defining the structure of the data sent and received by the API.
# Helping in Data Validation

# from_attributes = True, means that the schema can be populated even with object attributes, not only dictionaries.

## request: User registration validation
class UserCreate(BaseModel):
    email: EmailStr # EmailStr ensures automatic email validation
    password: str

## request: User login validation
class UserLogin(BaseModel):
    email: EmailStr
    password: str

## response: user output
class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config: 
        orm_mode = True # helps in serializing SQLAlchemy models to JSON (not possible by SQLAlchemy by default)

## response: after registering
class RegisterResponse(BaseModel):
    message: str
    newUser: UserOut

## response
class ArticleReferenceSchema(BaseModel):
    reference_id: int
    article_id: int
    source_url: Optional[str] = None
    published_date: Optional[datetime] = None
    source_name: Optional[str] = None
    bias: Optional[str] = None

    class Config:
        from_attributes = True

## response
class ArticleSchema(BaseModel):
    article_id: int
    title: str
    summary: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    left_bias: Optional[int] = None
    right_bias: Optional[int] = None
    center_bias: Optional[int] = None
    created_at: datetime
    views: int
    sources: List[ArticleReferenceSchema] = []  

    class Config:
        from_attributes = True
