from fastapi import APIRouter, HTTPException, Depends

router = APIRouter()

## 1. user-login endpoint
## user sends their email & password
## once the user logs in, JWT token is sent back as response and stored as cookie.

@router.post("/login")
def login(email: str, password: str):
    # if details were not sent
    if not email or not password:
        raise HTTPException(status_code=400, details="Either email or password was not sent")

    # retrieve user from db
    