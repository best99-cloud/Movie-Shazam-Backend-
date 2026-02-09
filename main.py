from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import requests
import os
import numpy as np

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-large-patch14"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Demo database (we will expand this later)
MOVIE_DB = [
    {"title": "Inception", "type": "Movie", "where": "Netflix", "vector": None},
    {"title": "Breaking Bad", "type": "TV Show", "where": "Netflix", "vector": None},
    {"title": "Naruto", "type": "Anime", "where": "Crunchyroll", "vector": None},
]

def get_embedding(image_bytes):
    r = requests.post(MODEL_URL, headers=HEADERS, data=image_bytes)
    emb = r.json()
    return np.array(emb).mean(axis=0)

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/")
def home():
    return {"status": "Movie Shazam AI running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_vec = get_embedding(image_bytes)

    best = None
    best_score = -1

    for item in MOVIE_DB:
        if item["vector"] is None:
            item["vector"] = img_vec

        score = similarity(img_vec, item["vector"])
        if score > best_score:
            best_score = score
            best = item

    return {
        "match": best["title"],
        "type": best["type"],
        "where_to_watch": best["where"],
        "confidence": float(best_score)
    }
