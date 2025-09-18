from fastapi import APIRouter, HTTPException, Depends, Response
# APIRouter: enables the creation of backend routes to access endpoints
# HTTPException: enables the creation of exceptions to let the user know the kind of problem that took place when accesing a route
# Depends: Helps you inject dependencies within route functions. These dependencies are required by the route and fastAPI sets it up for you.

from sqlalchemy.orm import Session

from api.user.model.userModel import User
from schemas.schemas import RegisterResponse, UserCreate, UserLogin, UserOut
from util import auth
from config.database import get_db

router = APIRouter()

## 1. user-register endpoint
## user sends their email & password
## creates an entry in the database

@router.post("/register", response_model=RegisterResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    
    # check if the user already exists in the table
    existingUser = db.query(User).filter(User.email == user.email).first()

    if existingUser:
        raise HTTPException(status_code=400, detail="Email already registered")

    # create an entry in the users table in the database
    newUser = User(
        email = user.email,
        hashed_password = auth.hash_pwd(user.password),
    )

    db.add(newUser)
    db.commit()
    db.refresh(newUser)
    return {"message": "Successfully Registered", "newUser": newUser}


## 2. user-login endpoint
## user sends their email & password
## once the user logs in, JWT token is sent back as response and stored as cookie.

@router.post("/login")
def login(user: UserLogin, response: Response, db: Session = Depends(get_db)):

    # check if user actually exists
    existingUser = db.query(User).filter(User.email == user.email).first()
    if not existingUser:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # check if password is correct
    if not auth.verify_pwd(user.password, existingUser.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # create a JWT token
    token = auth.createAccessToken({
        "email" : user.email,
    }, 60 * 60)

    # set the cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False, # true for production builds, false for test builds
        samesite='lax',
        max_age=60*60,
    )

    return {"message" : "Sucessfully logged In!", "token": token}

    