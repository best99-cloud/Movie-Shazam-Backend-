from fastapi import FastAPI, UploadFile, File
import requests
import os
import numpy as np
import threading
import base64

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
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
# HuggingFace CLIP embedding (CORRECT FORMAT)
# -------------------------------
def get_embedding(image_bytes):
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "inputs": b64
        }

        r = requests.post(
            MODEL_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        r.raise_for_status()
        data = r.json()

        # HF returns [[[...]]] sometimes
        if isinstance(data, list):
            data = np.array(data).squeeze()

        return np.array(data, dtype=np.float32)

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
# Background preload
# -------------------------------
def prepare_database():
    for item in MOVIE_DB:
        try:
            img = requests.get(item["image"], timeout=20).content
            vec = get_embedding(img)
            item["vector"] = vec
            print("Loaded:", item["title"])
        except Exception as e:
            print("Failed loading", item["title"], e)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=prepare_database, daemon=True).start()

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
