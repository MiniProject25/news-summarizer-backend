# entry point

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return "Server running on PORT 8000"