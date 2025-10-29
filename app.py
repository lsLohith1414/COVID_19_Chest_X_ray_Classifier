from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse

from src.pipeline.training_pipeline import TrainingPipline
from src.entities.global_config import GlobalConfig
from src.pipeline.prediction_pipline import predict_image
from src.constants.model_training_constants import IMAGE_SIZE


from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn


app = FastAPI(title="Xray Classifier API")


@app.get("/")
def home():
    return {"message":"home page of the API"}


@app.get("/about")
def about():
    return {"message":"About page of the api"}


@app.get("/train")
def train_model():
    global_config = GlobalConfig()
    train_pipline = TrainingPipline(global_config=global_config)
    evaluation_artifact = train_pipline.run_training_pipline()

    return {
        "accuracy": evaluation_artifact.acc,
        "precision": evaluation_artifact.precision,
        "recall": evaluation_artifact.recall,
        "f1_score": evaluation_artifact.f1
    }




# Example constants -- replace with actual values or load from config
MODEL_PATH = "final_models/vgg19_final_model.keras"
CLASS_NAMES = ['COVID', 'Normal', 'Pneumonia']
TARGET_SIZE = IMAGE_SIZE



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # basic validation
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()  # read bytes from UploadFile
    try:
        result = predict_image(MODEL_PATH, image_bytes, CLASS_NAMES, target_size=TARGET_SIZE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return result
