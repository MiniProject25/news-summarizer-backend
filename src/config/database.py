## setting up a postgres database
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# DB Connection
load_dotenv()

database_uri = os.load_env("DATABASE_URI")
engine = create_engine(database_uri)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)