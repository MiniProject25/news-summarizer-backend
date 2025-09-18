from sqlalchemy import Column, Integer, String
from config.database import Base

# Object-Relational Mapper 
# User class maps directly to the "Users" table in the Database
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False) 