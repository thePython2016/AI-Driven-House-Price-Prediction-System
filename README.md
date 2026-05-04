#  House Price Prediction App

A machine learning web application that predicts house prices from uploaded CSV data, built with Streamlit and Scikit-learn.

🔗 Live App: https://myhousepredictionmodel.streamlit.app/

##  Model Performance

| Metric   | Score  |
| -------- | ------ |
| R² Score | 0.9015 |
| RMSE     | 0.1294 |

> The model explains ~90% of the variance in house prices, with low prediction error.

##  Features

* 📂 Upload CSV file with house features (predefined format)
* 🔍 Instant predictions using trained ML model
* ✅ Prediction summary and feedback
* ⬇️ Download predictions as CSV

##  Model Details

* Algorithm: Scikit-learn regression model
* Preprocessing: ColumnTransformer + OneHotEncoding
* Input: Structured housing dataset
* Output: Predicted house prices



## ⚙️ Run Locally

### 1. Clone the repository

git clone https://github.com/thePython2016/studentScorePredictionModel.git

cd studentScorePredictionModel

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the app
streamlit run app.py

## 📥 Sample Data

Use the provided `Test.csv` to test the app format.

##  Requirements

See `requirements.txt` for full dependencies.


##  How to Use

1. Open the live app
2. Upload a CSV file (use sample format)
3. Click Predict button
4. View predicted prices
5. Download results as CSV


