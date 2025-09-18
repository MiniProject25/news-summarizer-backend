from pydantic import BaseModel, EmailStr

# These schemas help defining the structure of the data sent and received by the API.
# Helping in Data Validation

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

class RegisterResponse(BaseModel):
    message: str
    newUser: UserOut