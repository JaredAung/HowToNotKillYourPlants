# main.py
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env - try project root, then search upward from cwd
_env_paths = [
    Path(__file__).resolve().parent.parent / ".env",  # project root
    find_dotenv(usecwd=True),
    find_dotenv(usecwd=False),
]
for _p in _env_paths:
    if _p and Path(_p).exists():
        load_dotenv(_p, override=True)
        break

from auth.auth import router as auth_router
from chat.chat_router import router as chat_router
from garden.garden import router as garden_router
from plant.plant import router as plant_router
from profile.profile import router as profile_router
from recommend.recommend import router as recommend_router
from search.search import router as search_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(garden_router)
app.include_router(plant_router)
app.include_router(profile_router)
app.include_router(recommend_router)
app.include_router(search_router)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
