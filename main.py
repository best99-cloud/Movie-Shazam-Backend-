from fastapi import FastAPI, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Movie Shazam backend running"}

@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    return {
        "filename": file.filename,
        "format": image.format,
        "size": image.size
    }
