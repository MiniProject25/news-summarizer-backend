# entry point

from fastapi import FastAPI 
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from middlewares.middleware import AuthMiddleware

load_dotenv()

from config.database import engine, Base
from api.index import router as api_router

# create all tables defined in userModel (defined by SQLAlchemy)

origins = [
    "http://localhost:3000"
]

# applying rate limit to the backend

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return "Server running on PORT 8000"