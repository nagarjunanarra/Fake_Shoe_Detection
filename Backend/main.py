# Backend/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from model import predict_image  

app = FastAPI()

# Enable CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    API endpoint to handle image uploads and return predictions.
    """
    if file.content_type.startswith("image/"):
        try:
            image_data = await file.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")  

            # Get prediction from the model
            result = predict_image(image)
            return result

        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
    else:
        return {"error": "Invalid file format. Please upload an image."}


@app.get("/")
def read_root():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"message": "API is running. Use the /upload endpoint to upload images for prediction."}
