from fastapi import FastAPI, UploadFile, File
import requests
import os
import numpy as np
import base64

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-large-patch14"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -------------------------------
# Reference images (POSTERS / SCENES)
# -------------------------------
MOVIE_DB = [
    {
        "title": "Inception",
        "type": "Movie",
        "where": "Netflix",
        "image": "https://upload.wikimedia.org/wikipedia/en/7/7f/Inception_ver3.jpg",
        "vector": None
    },
    {
        "title": "Breaking Bad",
        "type": "TV Show",
        "where": "Netflix",
        "image": "https://upload.wikimedia.org/wikipedia/en/6/61/Breaking_Bad_title_card.png",
        "vector": None
    },
    {
        "title": "Naruto",
        "type": "Anime",
        "where": "Crunchyroll",
        "image": "https://upload.wikimedia.org/wikipedia/en/9/94/NarutoCoverTankobon1.jpg",
        "vector": None
    },
]

# -------------------------------
# CLIP embedding via HuggingFace
# -------------------------------
def get_embedding(image_bytes):
    response = requests.post(
        MODEL_URL,
        headers=HEADERS,
        data=image_bytes,
    )

    data = response.json()

    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])

    return np.array(data).mean(axis=0)

# -------------------------------
# Cosine similarity
# -------------------------------
def similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------------------
# Pre-embed reference images
# -------------------------------
def prepare_database():
    for item in MOVIE_DB:
        if item["vector"] is None:
            print("Embedding:", item["title"])
            img = requests.get(item["image"]).content
            item["vector"] = get_embedding(img)

# Run once on startup
prepare_database()

# -------------------------------
# Routes
# -------------------------------
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
        score = similarity(img_vec, item["vector"])
        if score > best_score:
            best_score = score
            best = item

    return {
        "match": best["title"],
        "type": best["type"],
        "where_to_watch": best["where"],
        "confidence": round(best_score, 4)
    }
