from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import numpy as np

app = FastAPI()

model = None
preprocess = None
index = None
metadata = []

def load_ai():
    global model, preprocess, index
    if model is None:
        import torch
        import clip
        import faiss
        model, preprocess = clip.load("ViT-B/32")
        index = faiss.IndexFlatL2(512)

@app.get("/")
def home():
    return {"status": "Movie Shazam AI ready"}

@app.post("/add-frame")
async def add_frame(file: UploadFile, movie: str, timestamp: str):
    load_ai()
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = preprocess(image).unsqueeze(0)

    import torch
    with torch.no_grad():
        vec = model.encode_image(img).numpy()

    index.add(vec)
    metadata.append({"movie": movie, "timestamp": timestamp})
    return {"status": "frame added"}

@app.post("/search")
async def search(file: UploadFile):
    load_ai()
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = preprocess(image).unsqueeze(0)

    import torch
    with torch.no_grad():
        vec = model.encode_image(img).numpy()

    D, I = index.search(vec, 1)
    return metadata[I[0][0]]
