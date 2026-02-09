from fastapi import FastAPI, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Movie Shazam backend running (AI loading lazily)"}

@app.post("/upload")
async def upload(file: UploadFile):
    data = await file.read()
    img = Image.open(io.BytesIO(data))
    return {"width": img.width, "height": img.height}
