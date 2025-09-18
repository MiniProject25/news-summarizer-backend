# util functions for authentication
## necessary libraries
from fastapi import HTTPException
from passlib.context import CryptContext # for hashing & verifying passwords
from jose import ExpiredSignatureError, JWTError, jwt # for creating jwt tokens
from datetime import datetime, timedelta # for setting expiry time for the jwt token
import os # for accessing environment variables

# creating a "context" that knows what hashing alg to apply (in our case, it is bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

### hash password
def hash_pwd(password: str) -> str:
    return pwd_context.hash(password) 

### verify password util
def verify_pwd(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

### creating an access token
def createAccessToken(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()

    # setting expiry 
    if (expires_delta):
        expire = datetime.now() + timedelta(minutes=expires_delta)
    else:
        expire = datetime.now() + timedelta(minutes=os.getenv("DEF_EXPIRY")) # expiry time

    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, os.getenv("JWT_SECRET"), algorithm="HS256")

### for verifying access tokens
def verifyAccessToken(token: str) -> dict:
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        return payload
    except (ExpiredSignatureError, JWTError):   
        raise HTTPException(status_code=401, detail="Token invalid or expired")