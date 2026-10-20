
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# 1. DATA PREPARATION
# Membaca dataset Bank Marketing dari UCI
df = pd.read_csv("bank-full.csv", sep=";")

print("Jumlah data:", df.shape)
print("\nContoh data:")
print(df.head())

# 2. DEFINISI FITUR (X) DAN TARGET (y)
# Target: yes = berlangganan, no = tidak berlangganan
X = df.drop(columns=["y", "duration"])
y = df["y"].map({"no": 0, "yes": 1})

# Kolom kategorikal dan numerik
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

# 3. PREPROCESSING
# Mengubah data kategorikal menjadi angka
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numerical", "passthrough", numerical_features)
    ]
)

# 4. TRAIN-TEST SPLIT (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. MODELING: RANDOM FOREST
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 6. EVALUASI MODEL
y_pred = model.predict(X_test)

print("\n--- HASIL PREDIKSI PEMASARAN BANK ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Tidak Berlangganan", "Berlangganan"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. FEATURE IMPORTANCE
# Mengetahui fitur yang paling berpengaruh
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    model.named_steps["classifier"]
    .feature_importances_
)

feature_importances = pd.Series(
    importances,
    index=feature_names
).sort_values(ascending=False)

print("\nFitur Paling Berpengaruh:")
print(feature_importances.head(10))

# 8. VISUALISASI FITUR TERPENTING
feature_importances.head(10).sort_values().plot(
    kind="barh",
    title="10 Fitur Paling Berpengaruh"
)

plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# 9. CONTOH PREDIKSI NASABAH
hasil = pd.DataFrame({
    "Aktual": y_test,
    "Prediksi": y_pred
})

hasil["Keterangan"] = hasil["Prediksi"].map({
    0: "Tidak Berlangganan",
    1: "Berlangganan"
})

print("\nContoh Hasil Prediksi:")
print(hasil.head(10))
