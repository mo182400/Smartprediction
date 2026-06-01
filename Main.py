import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# STEP 1: Load training data
# -----------------------------
print("✅ Loading training data...")
train_df = pd.read_csv("training.csv")

# -----------------------------
# STEP 2: Clean training data
# -----------------------------
print("✅ Cleaning training data...")

# Remove rows where Code is missing
train_df = train_df.dropna(subset=["Code"])

# Fill missing Description
train_df["Description"] = train_df["Description"].fillna("")

# Ensure Code is integer
train_df["Code"] = train_df["Code"].astype(int)

print("\n🔍 Training data check:")
print(train_df.isna().sum())

# -----------------------------
# STEP 3: Train model
# -----------------------------
print("✅ Training model...")

vectorizer = TfidfVectorizer()
model = LogisticRegression(max_iter=1000)

X_train = train_df["Description"]
y_train = train_df["Code"]

X_train_vec = vectorizer.fit_transform(X_train)
model.fit(X_train_vec, y_train)

# -----------------------------
# STEP 4: Load input Excel
# -----------------------------
print("✅ Loading Excel file...")
input_df = pd.read_excel("input.xlsx")

# Normalize column names
input_df.columns = input_df.columns.str.strip().str.lower()

print("📊 Columns found in Excel:", list(input_df.columns))

# Try to find description column using keywords
keywords = ["description", "desc", "details", "narration", "text"]

desc_col = None
for col in input_df.columns:
    for key in keywords:
        if key in col:
            desc_col = col
            break
    if desc_col:
        break

# Fallback: use first column
if desc_col is None:
    desc_col = input_df.columns[0]
    print(f"⚠️ No matching column found. Using first column: {desc_col}")
else:
    print(f"✅ Using column: {desc_col}")

# Fill missing values
input_df[desc_col] = input_df[desc_col].fillna("")

# -----------------------------
# STEP 5: Predict
# -----------------------------
print("✅ Predicting codes...")

X_input = vectorizer.transform(input_df[desc_col])

# Predictions
input_df["PredictedCode"] = model.predict(X_input)

# Confidence score
probs = model.predict_proba(X_input)
input_df["Confidence"] = probs.max(axis=1)

# -----------------------------
# STEP 6: Save output
# -----------------------------
print("\n🎯 RESULT:\n")
print(input_df)

output_file = "output.xlsx"
input_df.to_excel(output_file, index=False)

print(f"\n✅ Output saved as {output_file}")
