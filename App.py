from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import os
import uuid

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# -----------------------------
# Setup folders
# -----------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Train model on startup
# -----------------------------
print("✅ Loading training data...")
train_df = pd.read_csv("training.csv")

print("✅ Cleaning training data...")
train_df = train_df.dropna(subset=["Code"])
train_df["Description"] = train_df["Description"].fillna("")
train_df["Code"] = train_df["Code"].astype(int)

print("✅ Training model...")
vectorizer = TfidfVectorizer()
model = LogisticRegression(max_iter=1000)

X_train = train_df["Description"]
y_train = train_df["Code"]

X_train_vec = vectorizer.fit_transform(X_train)
model.fit(X_train_vec, y_train)

print("✅ Model ready!")

# -----------------------------
# Upload endpoint
# -----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_id = str(uuid.uuid4())

    input_path = f"{UPLOAD_DIR}/{file_id}.xlsx"
    output_path = f"{OUTPUT_DIR}/{file_id}_output.xlsx"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Read Excel
    df = pd.read_excel(input_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Detect description column
    keywords = ["description", "desc", "details", "narration", "text"]

    desc_col = None
    for col in df.columns:
        for key in keywords:
            if key in col:
                desc_col = col
                break
        if desc_col:
            break

    # Fallback if no match
    if desc_col is None:
        desc_col = df.columns[0]

    print(f"✅ Using column: {desc_col}")

    # Clean text
    df[desc_col] = df[desc_col].fillna("")

    # Predict
    X_input = vectorizer.transform(df[desc_col])
    df["PredictedCode"] = model.predict(X_input)

    probs = model.predict_proba(X_input)
    df["Confidence"] = probs.max(axis=1)

    # Save output
    df.to_excel(output_path, index=False)

    return {
        "message": "done",
        "download_url": f"/download/{file_id}"
    }

# -----------------------------
# Download endpoint
# -----------------------------
@app.get("/download/{file_id}")
def download_file(file_id: str):
    file_path = f"{OUTPUT_DIR}/{file_id}_output.xlsx"

    return FileResponse(
        path=file_path,
        filename="output.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -----------------------------
# Serve UI
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
