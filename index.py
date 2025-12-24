from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from io import BytesIO
import numpy as np
from services.geocode_service import get_coordinates
from services.weather_service import get_weather
from services.dosage_rules import load_remedies, adjust_dosage
from services.llm_service import generate_advice

# ---------------- APP ----------------
app = FastAPI(
    title="Rice Disease Detection & Advisory API",
    version="1.0.0"
)

# ---------------- MODEL ----------------
MODEL_PATH = "model/efficientnetb3_rice_optimized.h5"
IMG_SIZE = (300, 300)

CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "sheath_blight",
    "tungro_virus"
]

model = load_model(MODEL_PATH)


@app.post("/predict-and-remedy")
async def predict_and_remedy(
    image_file: UploadFile = File(...),
    location_name: str = Form(...)
):
    # 1️⃣ Validate image
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed"
        )

    try:
        # ================= IMAGE PREDICTION =================
        contents = await image_file.read()
        img = image.load_img(
            BytesIO(contents),
            target_size=IMG_SIZE
        )

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)[0]
        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id])

        disease_key = CLASS_NAMES[class_id]

        # If healthy → stop early
        if disease_key == "healthy":
            return {
                "status": "success",
                "disease": "Healthy Crop 🌱",
                "confidence": round(confidence, 4),
                "message": "Your crop is healthy. No treatment required."
            }

        # ================= REMEDY LOGIC =================
        remedies = load_remedies()

        if disease_key not in remedies:
            raise HTTPException(
                status_code=404,
                detail="Remedy not found for predicted disease"
            )

        disease_data = remedies[disease_key]

        lat, lon, location = get_coordinates(location_name)
        weather = get_weather(lat, lon)

        base_dosage = float(
            disease_data["pesticide"]["dosage"].split()[0]
        )

        final_dosage = adjust_dosage(
            base_dosage,
            weather["humidity"],
            weather["rainfall"]
        )

        context = {
            "disease": disease_data["disease"],
            "location": location,
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "rainfall": weather["rainfall"],
            "chemical": disease_data["pesticide"]["chemical"],
            "dosage": f"{final_dosage} g/ml per liter",
            "organic": disease_data["organic_remedies"],
            "safety": disease_data["safety_rules"]
        }

        advice = generate_advice(context)

        # ================= FINAL RESPONSE =================
        return {
            "status": "success",
            "prediction": {
                "disease_key": disease_key,
                "confidence": round(confidence, 4)
            },
            "data": context,
            "advice": advice
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
