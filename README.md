# Mobile_Price_prediction(ML) \
A machine learning project that estimates the price of a smartphone from its specifications. It uses a Random Forest Regressor trained on 1,370 phones and comes with a Streamlit web app where we can enter specs and get an instant price estimate.

# Project Structure

mobile-price-prediction/
├── app.py                        # Streamlit web app \
├── price.ipynb                   # Data cleaning, training, tuning, evaluation \
├── mobilepriceprediction.csv     # Dataset (1,370 phones) \
├── models/ \
│   └── mobile_price_model.pkl    # Saved model + feature columns \
├── requirements.txt \
└── README.md 

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
10.Android_version 

# Algorithm Used And Steps Performed

**Algorithm:**  Random Forest Regressor. \
**Steps:** \
1.Loaded and cleaned the data. \
2.Extracted numeric specs and created a Brand column. \
3.One-hot encoded categories and filled missing values. \
4.Split into 80% train and 20% test. \
5.Tuned hyperparameters with RandomizedSearchCV and GridSearchCV. \
6.Evaluated the model (R² ≈ 0.74, MAE ≈ ₹9,300). \
7.Saved the model and built a Streamlit app.

# Tech Stack for this project

1.Language: Python \
2.Data handling: pandas, NumPy \
3.Machine learning: scikit-learn (RandomForestRegressor, RandomizedSearchCV, GridSearchCV) \
4.Model saving: joblib \
5.Web app: Streamlit \
6.Development: Jupyter Notebook \
7.Version control: Git and GitHub

# Run the application

1.Open VS Code \
2. File → Open Folder (select your project folder)  \
3. Terminal → New Terminal → run pip install -r requirements.txt  \  → run streamlit run app.py \
4. open http://localhost:8501  \
5.enter specs and click Predict price.

