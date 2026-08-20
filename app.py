from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

app = Flask(__name__)

# -----------------------------
# Load and prepare dataset
# -----------------------------
df = pd.read_csv("data.csv")

df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df = df.drop(columns=["date"])

# Keep the same categorical encoding approach as the original project.
city_categories = sorted(df["city"].astype(str).unique())
country_categories = sorted(df["country"].astype(str).unique())

city_map = {name: index for index, name in enumerate(city_categories)}
country_map = {name: index for index, name in enumerate(country_categories)}

df["city"] = df["city"].astype(str).map(city_map)
df["country"] = df["country"].astype(str).map(country_map)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

metrics = {
    "train_r2": round(float(r2_score(y_train, train_prediction)), 4),
    "test_r2": round(float(r2_score(y_test, test_prediction)), 4),
    "train_rmse": round(float(root_mean_squared_error(y_train, train_prediction)), 2),
    "test_rmse": round(float(root_mean_squared_error(y_test, test_prediction)), 2),
}

features = list(X.columns)

# USD conversion rate used only for displaying the prediction.
# Change this value if you want to use a different exchange rate.
USD_TO_INR = 88.0

# -----------------------------
# Field guidance for the form
# -----------------------------
# Several columns (waterfront, view, condition, etc.) use coded numbers that
# aren't obvious to someone filling in the form. This metadata drives the
# UI so each field either becomes a dropdown with plain-English labels, or
# a number input with a visible min/max range and a short hint, instead of
# a bare "Enter value" box.
CURRENT_YEAR = pd.Timestamp.now().year

FEATURE_META = {
    "bedrooms": {
        "kind": "number", "min": 0, "max": 10, "step": 1,
        "hint": "Whole number, 0-10",
    },
    "bathrooms": {
        "kind": "number", "min": 0, "max": 8, "step": 0.25,
        "hint": "0-8, in quarters (e.g. 1.5, 2.25)",
    },
    "sqft_living": {
        "kind": "number", "min": 300, "max": 14000, "step": 10,
        "hint": "Living area, roughly 300-14,000 sqft",
    },
    "sqft_lot": {
        "kind": "number", "min": 500, "max": 1100000, "step": 10,
        "hint": "Total lot size, roughly 500-1,100,000 sqft",
    },
    "floors": {
        "kind": "select",
        "options": [
            {"value": "1", "label": "1 floor"},
            {"value": "1.5", "label": "1.5 floors"},
            {"value": "2", "label": "2 floors"},
            {"value": "2.5", "label": "2.5 floors"},
            {"value": "3", "label": "3 floors"},
            {"value": "3.5", "label": "3.5 floors"},
        ],
        "hint": "Number of floors in the house",
    },
    "waterfront": {
        "kind": "select",
        "options": [
            {"value": "0", "label": "No"},
            {"value": "1", "label": "Yes"},
        ],
        "hint": "Does the property have waterfront access?",
    },
    "view": {
        "kind": "select",
        "options": [
            {"value": "0", "label": "0 - None"},
            {"value": "1", "label": "1 - Fair"},
            {"value": "2", "label": "2 - Average"},
            {"value": "3", "label": "3 - Good"},
            {"value": "4", "label": "4 - Excellent"},
        ],
        "hint": "How good is the view from the property?",
    },
    "condition": {
        "kind": "select",
        "options": [
            {"value": "1", "label": "1 - Poor"},
            {"value": "2", "label": "2 - Fair"},
            {"value": "3", "label": "3 - Average"},
            {"value": "4", "label": "4 - Good"},
            {"value": "5", "label": "5 - Excellent"},
        ],
        "hint": "Overall condition of the property",
    },
    "sqft_above": {
        "kind": "number", "min": 300, "max": 9500, "step": 10,
        "hint": "Living area above ground, roughly 300-9,500 sqft",
    },
    "sqft_basement": {
        "kind": "number", "min": 0, "max": 5000, "step": 10,
        "hint": "Basement area in sqft. Enter 0 if there's no basement",
    },
    "yr_built": {
        "kind": "number", "min": 1900, "max": CURRENT_YEAR, "step": 1,
        "hint": f"Year built, 1900-{CURRENT_YEAR}",
    },
    "yr_renovated": {
        "kind": "number", "min": 0, "max": CURRENT_YEAR, "step": 1,
        "hint": f"Year renovated. Enter 0 if never renovated",
    },
    "day": {
        "kind": "number", "min": 1, "max": 31, "step": 1,
        "hint": "Day of sale, 1-31",
    },
    "month": {
        "kind": "number", "min": 1, "max": 12, "step": 1,
        "hint": "Month of sale, 1-12",
    },
    "year": {
        "kind": "number", "min": 1900, "max": CURRENT_YEAR, "step": 1,
        "hint": f"Year of sale, 1900-{CURRENT_YEAR}",
    },
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        metrics=metrics,
        cities=city_categories,
        countries=country_categories,
        features=features,
        feature_meta=FEATURE_META,
        rows=len(df),
        usd_to_inr=USD_TO_INR,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        values = []

        for feature in features:
            value = data.get(feature)

            if value is None or value == "":
                raise ValueError(f"Please enter {feature.replace('_', ' ')}")

            if feature == "city":
                if str(value) not in city_map:
                    raise ValueError("Please select a valid city.")
                value = city_map[str(value)]

            elif feature == "country":
                if str(value) not in country_map:
                    raise ValueError("Please select a valid country.")

                value = country_map[str(value)]

            else:
                value = float(value)

                meta = FEATURE_META.get(feature)
                if meta and meta["kind"] == "number":
                    if value < meta["min"] or value > meta["max"]:
                        raise ValueError(
                            f"{feature.replace('_', ' ').title()} should be between "
                            f"{meta['min']} and {meta['max']}."
                        )

            values.append(value)

        prediction = float(model.predict(np.array(values).reshape(1, -1))[0])
        usd_prediction = prediction / USD_TO_INR

        return jsonify({
            "success": True,
            "inr": round(prediction, 2),
            "usd": round(usd_prediction, 2),
            "model": "Linear Regression"
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
