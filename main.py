from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import torch
import clip
import faiss
import numpy as np

app = FastAPI()

model, preprocess = clip.load("ViT-B/32")

index = faiss.IndexFlatL2(512)
metadata = []

@app.get("/")
def home():
    return {"status": "Movie Shazam AI running"}

@app.post("/add-frame")
async def add_frame(file: UploadFile, movie: str, timestamp: str):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vec = model.encode_image(img).numpy()

    index.add(vec)
    metadata.append({
        "movie": movie,
        "timestamp": timestamp
    })

    return {"status": "frame added"}

@app.post("/search")
async def search(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vec = model.encode_image(img).numpy()

    if index.ntotal == 0:
        return {"error": "No movies indexed yet"}

    D, I = index.search(vec, 1)
    return metadata[I[0][0]]
