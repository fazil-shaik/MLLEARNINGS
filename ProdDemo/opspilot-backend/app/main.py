from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.database.connection import init_schema
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_schema()  # idempotent — safe to run every startup
    yield


app = FastAPI(title="OpsPilot", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
