from fastapi import APIRouter
from api.articles.controller import articleController

router = APIRouter()

router.include_router(articleController.router, prefix="", tags=["articles"])