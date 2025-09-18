## setting up a postgres database
from sqlalchemy import create_engine # function that helps in creating the connection to the postgres database
from sqlalchemy.ext.declarative import declarative_base # factory function that helps in creating base class for database models to inherit from
from sqlalchemy.orm import sessionmaker # for creating "sessions" that help in talking to our database
import os # accessing environment variables

# DB Connection
print("Database URI: ", os.getenv("DATABASE_URI"))
database_uri = os.getenv("DATABASE_URI")
engine = create_engine(database_uri)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()