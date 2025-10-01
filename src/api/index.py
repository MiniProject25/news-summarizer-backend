from fastapi import APIRouter
from api.user.index import router as userRouter
from api.articles.index import router as articleRouter

router = APIRouter()
router.include_router(userRouter, prefix="/user")
router.include_router(articleRouter, prefix="/article")
