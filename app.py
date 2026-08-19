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


@app.route("/")
def home():
    return render_template(
        "index.html",
        metrics=metrics,
        cities=city_categories,
        countries=country_categories,
        features=features,
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
