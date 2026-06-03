import os
import re
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# -----------------------------
# SETUP
# -----------------------------
app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

# -----------------------------
# LOAD & TRAIN MODEL
# -----------------------------
def load_data():
    df = pd.read_csv("training.csv")

    if os.path.exists("feedback.csv"):
        fb = pd.read_csv("feedback.csv")
        df = pd.concat([df, fb], ignore_index=True)

    df = df.dropna(subset=["Code"])
    df["Description"] = df["Description"].fillna("").apply(clean_text)
    df["Code"] = df["Code"].astype(int)

    return df

train_df = load_data()

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=5000
    )),
    ("clf", LinearSVC())
])

model.fit(train_df["Description"], train_df["Code"])
print("✅ Model ready")

# -----------------------------
# RULE ENGINE
# -----------------------------
def apply_rules(text):
    text = text.lower()

    if "tax" in text:
        return 500, 0.99
    if "salary" in text:
        return 300, 0.99
    if "refund" in text:
        return 200, 0.95

    return None, None

# -----------------------------
# PREDICT
# -----------------------------
def predict(text):

    text_clean = clean_text(text)

    code, conf = apply_rules(text_clean)
    if code:
        return code, conf

    pred = model.predict([text_clean])[0]
    return int(pred), 0.75

# -----------------------------
# UPLOAD API
# -----------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    file_id = file.filename.replace(" ", "_")

    input_path = f"uploads/{file_id}"
    output_path = f"outputs/output_{file_id}"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_excel(input_path)
    df.columns = df.columns.str.strip().str.lower()

    desc_col = None
    for col in df.columns:
        if "desc" in col:
            desc_col = col
            break

    if desc_col is None:
        desc_col = df.columns[0]

    df[desc_col] = df[desc_col].fillna("")

    codes = []
    confs = []

    for text in df[desc_col]:
        c, conf = predict(text)
        codes.append(c)
        confs.append(round(conf, 2))

    df["PredictedCode"] = codes
    df["Confidence"] = confs
    df["LowConfidence"] = df["Confidence"] < 0.7

    df.to_excel(output_path, index=False)

    return {
        "download_url": f"/download/output_{file_id}",
        "data": df.to_dict(orient="records")
    }

# -----------------------------
# DOWNLOAD
# -----------------------------
@app.get("/download/{file_name}")
def download(file_name: str):
    return FileResponse(f"outputs/{file_name}", filename=file_name)

# -----------------------------
# FEEDBACK
# -----------------------------
@app.post("/feedback")
async def save_feedback(data: dict = Body(...)):

    desc = data.get("description")
    code = data.get("code")

    fb = pd.DataFrame([[desc, code]], columns=["Description", "Code"])

    if os.path.exists("feedback.csv"):
        fb.to_csv("feedback.csv", mode="a", header=False, index=False)
    else:
        fb.to_csv("feedback.csv", index=False)

    return {"status": "saved"}

# -----------------------------
# RETRAIN
# -----------------------------
@app.post("/retrain")
def retrain():
    global model

    df = load_data()
    model.fit(df["Description"], df["Code"])

    return {"status": "retrained"}

# -----------------------------
# UI
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
