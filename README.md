# Medicine Price & Generic Alternative Finder

Predicts a fair price for a medicine based on its composition and manufacturer, and finds cheaper alternatives with the same active ingredients.

## Live demo
https://gdg-rec-aiml.onrender.com/

## Data
Indian Pharmaceutical Products dataset (Kaggle). Used a random sample of 8,000 rows for development instead of the full 250k+.

## Composition cleaning
`active_ingredients` comes in as a stringified list of `{name, strength}` pairs, so it needs to be parsed first. Ingredient names are lowercased, slashes/hyphens stripped, extra whitespace collapsed. Strengths are lowercased with spaces removed ("500 mg" -> "500mg"). For combination drugs, ingredients are sorted alphabetically before joining so order in the original data doesn't matter. This produces `cleaned_composition`, used to match/group medicines across brands.

## Encoding
Manufacturer and composition are label-encoded (too many unique values for one-hot). Dosage form and therapeutic class are one-hot encoded.

## Model
Random Forest Regressor. Compared against Gradient Boosting, Random Forest performed better. Dropped rows with price <= 0, discontinued products, and prices above the 99th percentile before training (outliers were skewing the error).

MAE: ~₹62.5 on held-out test data (mean test price ~₹128).

## Run locally
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --app-dir .
Open http://127.0.0.1:8000/

## Structure
backend/    FastAPI app - /predict-price and /alternatives
model/      trained model + encoders (joblib)
data/       processed dataset used by the API
frontend/   search page
notebooks/  data cleaning + model training
