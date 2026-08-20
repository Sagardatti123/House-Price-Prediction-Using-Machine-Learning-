# House-Price-Prediction-Using-Multiple-Linear-Regression

Deployment Link: https://sagar-datti-house-price-prediction-2.onrender.com

# 🏠 House Price Prediction using Multiple Linear Regression

**🤖 Machine Learning • 📈 Multiple Linear Regression • 🐍 Python • 🌐 Flask**

A Machine Learning web application that predicts the estimated price of a house based on multiple features such as location, date-related information, and other property attributes.

---

## 🚀 Project Overview

**House Price Prediction** is a Machine Learning web application built using **Multiple Linear Regression**.

The model learns the relationship between multiple independent variables and the house price.

### 📊 Input Features

* 🏙️ **City**
* 🌍 **Country**
* 📅 **Day**
* 📆 **Month**
* 🗓️ **Year**
* 🏠 Other numerical features available in the dataset

### 🎯 Target Variable

**House Price**

The dataset is loaded using Pandas, the date column is converted into separate **day, month and year** features, and the original date column is removed.

The model is trained using **Linear Regression** from Scikit-learn.

---

## 🧠 Machine Learning Model

This project uses **Multiple Linear Regression**.

### Linear Regression Equation

```text
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ... + βₙXₙ
```

Where:

* `Y` → Predicted House Price
* `β₀` → Intercept
* `β₁...βₙ` → Model coefficients
* `X₁...Xₙ` → Input features

The model learns the relationship between the available house features and the target house price.

---

## ✨ Features

* 🏠 House price prediction
* 📈 Multiple Linear Regression
* 🐍 Python-based Machine Learning
* 📊 Pandas data preprocessing
* 🔢 NumPy numerical operations
* 🌍 City and country encoding
* 📅 Date feature extraction
* 📈 R² score evaluation
* 📉 RMSE evaluation
* 🌐 Flask web application
* 💰 Price displayed in INR
* 💵 Price converted and displayed in USD
* 🚀 Deployment-ready Flask application
* ⚡ Real-time prediction through API

---

## 🛠️ Tech Stack

| Technology             | Purpose                |
| ---------------------- | ---------------------- |
| 🐍 Python              | Programming language   |
| 🤖 Scikit-learn        | Machine Learning       |
| 📊 Pandas              | Data processing        |
| 🔢 NumPy               | Numerical operations   |
| 📈 Linear Regression   | House price prediction |
| 🌐 Flask               | Web application        |
| 🎨 HTML/CSS/JavaScript | User interface         |
| 🚀 Gunicorn            | Production server      |
| 📁 CSV                 | Dataset storage        |

## The project's requirements include Flask, Pandas, NumPy, Scikit-learn and Gunicorn.

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── app.py                  # Flask application
├── main.py                 # Machine Learning training code
├── data.csv                # House price dataset
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment configuration
│
├── templates/
│   └── index.html          # Interactive web interface
│
└── README.md               # Project documentation
```

---

## 🔄 How It Works

```text
                🏠 House Dataset
                       │
                       ▼
              ┌─────────────────┐
              │  Data Cleaning  │
              └────────┬────────┘
                       │
                       ▼
             📅 Feature Engineering
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Day / Month / Year   City / Country
             │                   │
             └─────────┬─────────┘
                       ▼
              📊 Train/Test Split
                       │
                       ▼
             🤖 Linear Regression
                       │
                       ▼
                📈 Prediction
                       │
              ┌────────┴────────┐
              ▼                 ▼
          🇮🇳 INR Price      🇺🇸 USD Price
```

---

## 📊 Data Preprocessing

The project performs the following preprocessing steps.

### 1️⃣ Date Conversion

The original date column is converted into:

```text
Day
Month
Year
```

The original `date` column is then removed.

### 2️⃣ City Encoding

City values are converted into numerical values so they can be used by the Machine Learning model.

The Flask application creates a consistent mapping between city names and numerical values.

### 3️⃣ Country Encoding

Country values are also converted into numerical representations using a predefined mapping.

---

## 📚 Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

This allows the model to train on one portion of the dataset and evaluate its performance on unseen test data.

---

## 🤖 Model Training

The project uses Scikit-learn's `LinearRegression` algorithm.

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

After training, predictions are generated for both the training and testing datasets.

---

## 📈 Model Evaluation

The model is evaluated using two important metrics.

### 🎯 R² Score

R² Score measures how well the model explains the variation in house prices.

```text
Higher R² → Better model fit
```

The application calculates both:

```text
Training R²
Testing R²
```

### 📉 RMSE

RMSE stands for **Root Mean Squared Error**.

It measures the difference between the actual house prices and predicted house prices.

The application calculates:

```text
Training RMSE
Testing RMSE
```

These metrics are calculated automatically when the Flask application starts.

---

## 💰 Currency Conversion

The application displays the predicted house price in both **Indian Rupees** and **US Dollars**.

### 🇮🇳 INR

The original prediction is returned in INR.

### 🇺🇸 USD

The application uses the following conversion value:

```python
USD_TO_INR = 88.0
```

The USD prediction is calculated using:

```python
usd_prediction = prediction / USD_TO_INR
```

The exchange rate is currently configured as a project display value.

> ⚠️ The exchange rate can be changed in `app.py` if required.

---

## 🌐 Flask Application

The project uses Flask to create the web application.

### Home Route

```python
@app.route("/")
def home():
```

The home page loads the interactive `index.html` template and provides model metrics, cities, countries, features and dataset information to the frontend.

### Prediction Route

```python
@app.route("/predict", methods=["POST"])
def predict():
```

The `/predict` endpoint:

1. Receives user input.
2. Validates the values.
3. Encodes city and country.
4. Converts numerical values.
5. Sends the features to the trained model.
6. Generates the house price.
7. Converts the prediction to USD.
8. Returns the result as JSON.

---

## 🔌 Prediction API

### Endpoint

```text
POST /predict
```

### Prediction Process

```text
User Input
    ↓
JSON Request
    ↓
Input Validation
    ↓
City/Country Encoding
    ↓
Feature Preparation
    ↓
Linear Regression Model
    ↓
House Price
    ↓
INR + USD
```

### Response Format

```json
{
    "success": true,
    "inr": 8500000.0,
    "usd": 96590.91,
    "model": "Linear Regression"
}
```

The Flask endpoint returns the predicted INR amount, USD amount and model name when prediction succeeds.

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

### 3️⃣ Activate the Virtual Environment

#### Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

#### Windows CMD

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000/
```

Open the URL in your browser and enter the required house details.

---

## 🚀 Deployment

The project contains a `Procfile` and includes **Gunicorn** in its dependencies, making it suitable for deployment on platforms that support Python/Flask applications.

### Example Gunicorn Command

```text
web: gunicorn app:app
```

> ⚠️ Make sure the Gunicorn command matches the actual Flask filename and application variable in your repository.

---

## 🌐 Live Demo

### 🚀 Deployed Application

**Deployment Link:**

```text
https://house-price-prediction-using-machine-mjoi.onrender.com
```

Click the link above to try the House Price Prediction application.

---

## 💡 Example Prediction

Suppose the user enters all required house features:

```text
City       = Hyderabad
Country    = India
Day        = 19
Month      = 8
Year       = 2026
```

The remaining numerical features are entered through the web interface.

The model processes the values and generates:

```text
🏠 Predicted House Price

🇮🇳 INR: ₹XXXXXXXX
🇺🇸 USD: $XXXXXX
```

The exact prediction depends on the input values and the trained dataset.

---

## 🎯 Learning Objectives

This project demonstrates the complete Machine Learning workflow:

* ✅ Loading a dataset using Pandas
* ✅ Data cleaning
* ✅ Date preprocessing
* ✅ Feature engineering
* ✅ Categorical encoding
* ✅ Train-test splitting
* ✅ Multiple Linear Regression
* ✅ Model training
* ✅ Model prediction
* ✅ R² Score
* ✅ RMSE
* ✅ Flask integration
* ✅ REST API
* ✅ Currency conversion
* ✅ Web-based Machine Learning prediction

---

## 🔮 Future Improvements

* 📊 Add interactive prediction charts
* 📈 Add actual vs predicted price visualization
* 🤖 Compare Linear Regression with Random Forest
* 🤖 Add XGBoost Regression
* 📊 Add feature importance
* 💾 Save and load the trained model separately
* 📱 Improve responsive mobile design
* 🎨 Add advanced animations
* 📜 Add prediction history
* 📥 Add downloadable prediction reports
* 🌍 Add live currency exchange rates
* ☁️ Improve cloud deployment
* 🔐 Add user authentication

---

## ⚠️ Important Note

The current Flask application trains the Linear Regression model when the application starts and then uses that model for predictions.

For production deployment, a future improvement would be to train the model separately, save it using a serialization library such as Joblib, and load the trained model inside the Flask application.

---

## 👨‍💻 Author

### **Sagar Datti**

🎓 **B.Tech — Computer Science & Engineering (Data Science)**

💡 **Aspiring Data Scientist / Machine Learning Engineer**

### Interests

* 🤖 Machine Learning
* 📊 Data Science
* 🐍 Python
* 🧠 Artificial Intelligence
* 🌐 Flask
* 📈 Data Analytics

---

## ⭐ If You Like This Project

If this project helped you understand **Machine Learning, Multiple Linear Regression and Flask deployment**, consider giving this repository a ⭐.

Your support is appreciated! ❤️

---

<p align="center">

### 🏠 House Price Prediction

**Built with 🐍 Python + 🤖 Machine Learning + 🌐 Flask**

### 🚀 Project by **Sagar Datti**

⭐ **Don't forget to star the repository!** ⭐

</p>
