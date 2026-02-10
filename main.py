from fastapi import FastAPI, UploadFile, File
import requests
import os
import numpy as np
import threading
import base64

app = FastAPI()

# HuggingFace token (already set on Railway)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Correct CLIP endpoint
MODEL_URL = "https://router.huggingface.co/models/openai/clip-vit-large-patch14"

# -------------------------------
# Reference images (posters / frames)
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
    }
]

# -------------------------------
# HuggingFace CLIP embedding
# -------------------------------
def get_embedding(image_bytes):
    if not HF_TOKEN:
        print("HF_TOKEN missing")
        return None

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "inputs": {
            "image": b64
        }
    }

    try:
        r = requests.post(
            MODEL_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )

        if r.status_code != 200:
            print("HF error:", r.status_code, r.text)
            return None

        data = r.json()

        # HuggingFace sometimes returns nested arrays
        while isinstance(data, list):
            data = data[0]

        vec = np.array(data, dtype=np.float32)

        if vec.ndim > 1:
            vec = vec.mean(axis=0)

        return vec

    except Exception as e:
        print("Embedding exception:", e)
        return None

# -------------------------------
# Cosine similarity
# -------------------------------
def similarity(a, b):
    if a is None or b is None:
        return -1
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------------------
# Preload reference embeddings
# -------------------------------
def prepare_database():
    for item in MOVIE_DB:
        try:
            print("Embedding:", item["title"])
            img = requests.get(item["image"], timeout=15).content
            item["vector"] = get_embedding(img)
        except Exception as e:
            print("Failed:", item["title"], e)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=prepare_database).start()

# -------------------------------
# API routes
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
