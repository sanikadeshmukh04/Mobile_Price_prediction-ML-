# Mobile_Price_prediction(ML)

A machine learning project that estimates the price of a smartphone from its specifications. It uses a Random Forest Regressor trained on 1,370 phones and comes with a Streamlit web app where we can enter specs and get an instant price estimate.

# Project Structure

mobile-price-prediction/
├── app.py                        # Streamlit web app
├── price.ipynb                   # Data cleaning, training, tuning, evaluation
├── mobilepriceprediction.csv     # Dataset (1,370 phones)
├── models/
│   └── mobile_price_model.pkl    # Saved model + feature columns
├── requirements.txt
└── README.md
