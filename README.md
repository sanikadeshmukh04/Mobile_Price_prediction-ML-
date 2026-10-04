# Mobile_Price_prediction(ML)

A machine learning project that estimates the price of a smartphone from its specifications. It uses a Random Forest Regressor trained on 1,370 phones and comes with a Streamlit web app where we can enter specs and get an instant price estimate.

# Project Structure

mobile-price-prediction/
├── app.py                        # Streamlit web app \
├── price.ipynb                   # Data cleaning, training, tuning, evaluation \
├── mobilepriceprediction.csv     # Dataset (1,370 phones) \
├── models/ \
│   └── mobile_price_model.pkl    # Saved model + feature columns \
├── requirements.txt \
└── README.md \

# Features used for prediction

1.Brand \
2.Processor \
3.Ram_GB \
4.Battery_mAh \
5.Display_inch \
6.Main_Cam_MP \
7.Charge_W (fast charging) \
8.Ext_Mem_GB (external memory) \
9.Rating \
10.Android_version \
