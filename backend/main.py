import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Medicine Price & Generic Alternative Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, not per-request
model = joblib.load("model/price_model.pkl")
feature_cols = joblib.load("model/feature_columns.pkl")
df = pd.read_csv("data/medicines_processed.csv")


class PriceResponse(BaseModel):
    medicine_name: str
    predicted_price: float
    actual_price: float


class Alternative(BaseModel):
    brand_name: str
    manufacturer: str
    price_inr: float


class AlternativesResponse(BaseModel):
    medicine_name: str
    composition: str
    alternatives: list[Alternative]


def find_medicine(medicine_name: str):
    matches = df[df["brand_name"].str.lower().str.contains(medicine_name.lower(), na=False)]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_feature_row(row):
    values = {col: row[col] if col in row.index else 0 for col in feature_cols}
    return pd.DataFrame([values])[feature_cols]


# @app.get("/")
# def root():
#     return {"message": "Medicine Price & Generic Alternative Finder API"}


@app.get("/predict-price", response_model=PriceResponse)
def predict_price(medicine_name: str):
    row = find_medicine(medicine_name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No medicine found matching '{medicine_name}'")

    X = build_feature_row(row)
    predicted = model.predict(X)[0]

    return PriceResponse(
        medicine_name=row["brand_name"],
        predicted_price=round(float(predicted), 2),
        actual_price=round(float(row["price_inr"]), 2),
    )


@app.get("/alternatives", response_model=AlternativesResponse)
def get_alternatives(medicine_name: str, limit: int = 5):
    row = find_medicine(medicine_name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No medicine found matching '{medicine_name}'")

    composition = row["cleaned_composition"]
    if not composition:
        raise HTTPException(status_code=404, detail="Could not determine composition for this medicine")

    same_comp = df[df["cleaned_composition"] == composition]
    same_comp = same_comp[same_comp["brand_name"] != row["brand_name"]]
    same_comp = same_comp.sort_values("price_inr").head(limit)

    alternatives = [
        Alternative(
            brand_name=r["brand_name"],
            manufacturer=r["manufacturer"],
            price_inr=round(float(r["price_inr"]), 2),
        )
        for _, r in same_comp.iterrows()
    ]

    return AlternativesResponse(
        medicine_name=row["brand_name"],
        composition=composition,
        alternatives=alternatives,
    )

from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")