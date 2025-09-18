from fastapi import APIRouter
from api.user.controller import userController

router = APIRouter()
router.include_router(userController.router, prefix="", tags=["user"])