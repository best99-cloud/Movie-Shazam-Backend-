from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import clip
import numpy as np
import io
import threading

app = FastAPI()

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------------------------
# Movie database
# ---------------------------
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

# ---------------------------
# Encode image
# ---------------------------
def get_embedding(pil_image):
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]

# ---------------------------
# Cosine similarity
# ---------------------------
def similarity(a, b):
    return float(np.dot(a, b))

# ---------------------------
# Load reference images in background
# ---------------------------
def prepare_database():
    import requests
    for item in MOVIE_DB:
        try:
            img = Image.open(io.BytesIO(requests.get(item["image"]).content)).convert("RGB")
            item["vector"] = get_embedding(img)
            print("Loaded:", item["title"])
        except Exception as e:
            print("Failed:", item["title"], e)

@app.on_event("startup")
def startup():
    threading.Thread(target=prepare_database).start()

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return {"status": "Movie Shazam AI running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        query_vec = get_embedding(image)

        best = None
        best_score = -1

        for item in MOVIE_DB:
            if item["vector"] is None:
                continue
            score = similarity(query_vec, item["vector"])
            if score > best_score:
                best_score = score
                best = item

        return {
            "match": best["title"],
            "type": best["type"],
            "where_to_watch": best["where"],
            "confidence": round(best_score, 4)
        }

    except Exception as e:
        return {"error": str(e)}
