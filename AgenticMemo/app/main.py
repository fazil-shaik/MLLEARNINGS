from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Simple RAG API",
    description="A simple PDF RAG system using ChromaDB",
    version="1.0.0"
)


app.include_router(router)


@app.get("/")
def home():

    return {
        "message": "Simple RAG API is running"
    }