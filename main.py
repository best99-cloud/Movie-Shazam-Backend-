from fastapi import FastAPI, UploadFile, File
import requests
import os
import numpy as np
import threading

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-large-patch14"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -------------------------------
# Reference images
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
# Safe HuggingFace embedding
# -------------------------------
def get_embedding(image_bytes):
    try:
        r = requests.post(
            MODEL_URL,
            headers=HEADERS,
            data=image_bytes,
            timeout=30
        )
        r.raise_for_status()
        data = r.json()

        # HF sometimes returns [[[...]]]
        if isinstance(data, list) and isinstance(data[0], list):
            data = data[0]

        vec = np.array(data, dtype=np.float32)

        if len(vec.shape) > 1:
            vec = vec.mean(axis=0)

        return vec

    except Exception as e:
        print("Embedding error:", e)
        return None

# -------------------------------
# Cosine similarity
# -------------------------------
def similarity(a, b):
    if a is None or b is None:
        return -1
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------------------
# Background preload (DOES NOT BLOCK STARTUP)
# -------------------------------
def prepare_database():
    for item in MOVIE_DB:
        try:
            img = requests.get(item["image"], timeout=15).content
            vec = get_embedding(img)
            item["vector"] = vec
            print("Loaded:", item["title"])
        except Exception as e:
            print("Failed loading", item["title"], e)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=prepare_database).start()

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

    if img_vec is None:
        return {"error": "Failed to analyze image"}

    best = None
    best_score = -1

    for item in MOVIE_DB:
        score = similarity(img_vec, item["vector"])
        if score > best_score:
            best_score = score
            best = item

    if best is None:
        return {"error": "No match"}

    return {
        "match": best["title"],
        "type": best["type"],
        "where_to_watch": best["where"],
        "confidence": round(best_score, 4)
    }
