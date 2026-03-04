from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import traceback

app = FastAPI(title="SEHAT API", description="AI Plant Disease Detection")

# ── CORS – allow frontend from any origin ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = (224, 224)

# ── Load models once at startup ──
crop_model   = tf.keras.models.load_model("models/crop_classifier.h5")
potato_model = tf.keras.models.load_model("models/potato_model.h5")
tomato_model = tf.keras.models.load_model("models/tomato_model.h5")
apple_model  = tf.keras.models.load_model("models/apple_model.h5")
onion_model  = tf.keras.models.load_model("models/onion_model.h5")

# ── Load label maps ──
crop_labels   = json.load(open("models/crop_classifier_labels.json"))
potato_labels = json.load(open("models/potato_model_labels.json"))
tomato_labels = json.load(open("models/tomato_model_labels.json"))
apple_labels  = json.load(open("models/apple_model_labels.json"))
onion_labels  = json.load(open("models/onion_model_labels.json"))

# ── Load treatments ──
treatments = json.load(open("data/treatments.json"))

# ── Disease model routing ──
DISEASE_MODELS = {
    "potato": (potato_model, potato_labels),
    "tomato": (tomato_model, tomato_labels),
    "apple":  (apple_model,  apple_labels),
    "onion":  (onion_model,  onion_labels),
}


def preprocess(img_path: str) -> np.ndarray:
    """Load and preprocess image for model inference."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(img_path: str):
    """Run crop classification then disease detection. Returns (crop, disease, confidence)."""
    img = preprocess(img_path)

    # Crop classification
    crop_pred = crop_model.predict(img, verbose=0)
    crop_index = int(np.argmax(crop_pred))
    crop_name = crop_labels[str(crop_index)]

    # Route to disease model
    if crop_name not in DISEASE_MODELS:
        raise ValueError(f"Unknown crop detected: {crop_name}")

    model, labels = DISEASE_MODELS[crop_name]

    # Disease detection
    disease_pred = model.predict(img, verbose=0)
    disease_index = int(np.argmax(disease_pred))
    disease_name = labels[str(disease_index)]
    confidence = float(np.max(disease_pred))

    return crop_name, disease_name, confidence


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "SEHAT API is running"}


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html at root."""
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"status": "ok", "message": "SEHAT API is running. Frontend not found."}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Accept a leaf image and return crop, disease, treatment, and confidence."""

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are accepted.")

    path = f"temp_{file.filename}"

    try:
        # Save uploaded file
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run prediction
        crop, disease, confidence = predict(path)

        # Look up treatment
        treatment_data = treatments.get(
            disease,
            treatments.get(f"{crop}_{disease}", {})
        )
        treatment = treatment_data.get("treatment", [
            "No specific treatment plan found for this diagnosis.",
            "Please consult a local agricultural expert.",
            "Ensure proper soil health and watering."
        ])

        return {
            "status": "success",
            "crop": crop,
            "disease": disease,
            "treatment": treatment,
            "confidence": confidence
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Always clean up temp file
        if os.path.exists(path):
            os.remove(path)
