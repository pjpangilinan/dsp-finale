import os
import cv2
import numpy as np
import base64
import joblib
import tempfile
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_INPUT_SIZE = (64, 64)
DISPLAY_SIZE = (512, 512)
MODEL_DIR = "saved_models"

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    try:
        # Load models
        models["image"] = joblib.load(os.path.join(MODEL_DIR, "image_dsp.joblib"))
        models["video"] = joblib.load(os.path.join(MODEL_DIR, "video_dsp.joblib"))
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    yield
    models.clear()


app = FastAPI(title="AI-Generated Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# UTILS
# ==========================================

def encode_image_to_base64(image_array):
    success, buffer = cv2.imencode('.png', image_array)
    if not success: return None
    return base64.b64encode(buffer).decode('utf-8')


# ==========================================
# PIPELINES
# ==========================================

def process_image_pipeline(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None: return None, None, None

    # Model Input (64x64 Sobel)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, MODEL_INPUT_SIZE)
    sobel_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    features = magnitude_norm.flatten()

    # Visualization (High Res)
    gray_display = cv2.resize(gray, DISPLAY_SIZE)
    sx_disp = cv2.Sobel(gray_display, cv2.CV_64F, 1, 0, ksize=3)
    sy_disp = cv2.Sobel(gray_display, cv2.CV_64F, 0, 1, ksize=3)
    mag_disp = cv2.magnitude(sx_disp, sy_disp)
    mag_disp = cv2.normalize(mag_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(mag_disp, cv2.COLORMAP_JET)

    return features, img_bgr, heatmap


def process_video_pipeline(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None, None

    frames_model = []
    frames_display = []
    MAX_FRAMES = 30

    count = 0
    while count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_model.append(cv2.resize(gray, MODEL_INPUT_SIZE))
        frames_display.append(cv2.resize(gray, DISPLAY_SIZE))
        count += 1
    cap.release()

    if len(frames_model) < 2: return None, None, None

    # Model Features
    diffs_model = [cv2.absdiff(frames_model[i], frames_model[i - 1]) for i in range(1, len(frames_model))]
    var_map_model = np.var(np.array(diffs_model), axis=0)
    var_map_model = cv2.normalize(var_map_model, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    features = var_map_model.flatten()

    # Visualization
    diffs_disp = [cv2.absdiff(frames_display[i], frames_display[i - 1]) for i in range(1, len(frames_display))]
    var_map_disp = np.var(np.array(diffs_disp), axis=0)
    var_map_disp = cv2.normalize(var_map_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(var_map_disp, cv2.COLORMAP_INFERNO)

    return features, frames_display[0], heatmap


# ==========================================
# HELPER: PROBABILITY CALCULATION
# ==========================================
def get_ai_probability(model, features):
    features = features.reshape(1, -1)
    # predict_proba returns [[prob_class_0, prob_class_1]]
    # Index 0 = Real, Index 1 = Fake
    probs = model.predict_proba(features)[0]

    # We explicitly want the probability of it being FAKE (Index 1)
    ai_prob = probs[1]
    return float(ai_prob)


# ==========================================
# ROUTES
# ==========================================

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "AI-Generated Classifier API is running correctly.",
    }

@app.post("/analyze_image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    if "image" not in models: raise HTTPException(503, "Image Model not loaded.")

    features, img_orig, img_proc = process_image_pipeline(await file.read())
    if features is None: raise HTTPException(400, "Invalid image.")

    # --- NEW LOGIC ---
    ai_probability = get_ai_probability(models["image"], features)

    # Hard threshold: If AI probability is > 50%, call it AI.
    verdict = "AI-GENERATED" if ai_probability > 0.5 else "REAL"

    return JSONResponse(content={
        'original': encode_image_to_base64(img_orig),
        'processed': encode_image_to_base64(img_proc),
        'verdict': verdict,  # Explicitly "REAL" or "AI-GENERATED"
        'confidence': ai_probability,  # Returns 0.10 for Real, 0.90 for Fake
        'details': f"DSP Analysis (Sobel Gradient)",
        'explanation': (
            f"The system detected {ai_probability * 100:.1f}% likelihood of artificial manipulation based on gradient artifacts."
        )
    })


@app.post("/analyze_video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    if "video" not in models: raise HTTPException(503, "Video Model not loaded.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(await file.read())
        temp_filename = tfile.name

    try:
        features, img_orig, img_proc = process_video_pipeline(temp_filename)
        if features is None: raise HTTPException(400, "Invalid video.")

        # --- NEW LOGIC ---
        ai_probability = get_ai_probability(models["video"], features)

        verdict = "AI-GENERATED" if ai_probability > 0.5 else "REAL"

        return JSONResponse(content={
            'original': encode_image_to_base64(img_orig),
            'processed': encode_image_to_base64(img_proc),
            'verdict': verdict,
            'confidence': ai_probability,
            'details': f"DSP Analysis (Temporal Jitter)",
            'explanation': (
                f"The system detected {ai_probability * 100:.1f}% likelihood of artificial manipulation based on inter-frame variance."
            )
        })
    finally:
        if os.path.exists(temp_filename): os.unlink(temp_filename)


if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=8080)


