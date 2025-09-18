from fastapi import APIRouter
from api.user.index import router as userRouter

router = APIRouter()
router.include_router(userRouter, prefix="/user")