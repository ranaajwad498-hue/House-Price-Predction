import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

import joblib

class PricePrediction:

    def __init__(self, dataset):
        self.dataset_file = dataset
        self.raw_data = None
        self.df = None
        self.model = None

    def load_data(self):
        print("Loading Data.....")
        self.raw_data = pd.read_csv(self.dataset_file)
        print("\nFirst 5 Records:")
        print(self.raw_data.head())
        print("\nInformation of Dataset:")
        self.raw_data.info()

    def cleaning_data(self):
        print("\nCleaning Data.....")
        self.df = self.raw_data.copy()
        self.df.drop_duplicates(inplace=True)
        print("Filling Null Values.....")
        numerical_columns = self.df.select_dtypes(include=["number"]).columns
        for col in numerical_columns:
            self.df[col] = self.df[col].fillna( self.df[col].median() )

        categorical_columns = self.df.select_dtypes(include=["object", "string", "category"]).columns
        for col in categorical_columns:
             if not self.df[col].mode().empty:
                self.df[col] = self.df[col].fillna( self.df[col].mode()[0]  )

        self.df["Location"] = self.df["Location"].replace(["SUBURBS", "downtown"],["Suburbs", "Downtown"])
        self.df["Condition"] = self.df["Condition"].replace(["FAIR", "good"], ["Fair", "Good"] )
        self.df["Has_Pool"] = self.df["Has_Pool"].replace(["Y", "N", "TRUE", "FALSE"],["Yes", "No", "Yes", "No"] )
        self.df["Bathrooms"] = self.df["Bathrooms"].replace([1.5, 2.5, 3.5, 4.5],[2, 3, 4, 5]  )
        self.df["Area_SqFt"] = (self.df["Area_SqFt"] .astype(str) .str.extract(r"(\d+\.?\d*)")[0])
        self.df["Area_SqFt"] = pd.to_numeric( self.df["Area_SqFt"], errors="coerce")
        self.df["Area_SqFt"] = self.df["Area_SqFt"].fillna( self.df["Area_SqFt"].median())
        self.df["Bedrooms"] = self.df["Bedrooms"].clip(lower=0,upper=5)
        self.df["Price"] = self.df["Price"].clip( lower=0, upper=900000)
        print("\nChecking Null Values:")
        print(self.df.isnull().sum())
        self.df.drop(columns=["Id"], inplace=True)
        self.df.to_csv("Cleaned Data.csv",index=False)
        print("\nDataset Columns:")
        print(self.df.columns)

    def split_data(self):
        print("\nSplitting Data for Training and Testing.....")
        x = self.df.drop(columns=["Price"])
        y = self.df["Price"]
        self.xtrain,self.xtest,self.ytrain,self.ytest = train_test_split(x,y,test_size=0.2,random_state=42)
        print("Data Successfully Split")
        print("Training Data Shape:", self.xtrain.shape)
        print("Testing Data Shape:", self.xtest.shape)

    def fe(self):
        print("\nApplying Feature Engineering.....")

        numerical_columns = [
        "Area_SqFt","Bedrooms",
        "Bathrooms","Floors","Year_Built",
        "Garage_Capacity","Distance_City_Center_Miles"
    ]
        ordinal_columns = ["Condition"]
        nominal_columns = ["Location","Has_Pool"]

        condition_order = ["Poor","Fair","Good","Very Good","Excellent"]

        self.preprocessor = ColumnTransformer(transformers=[
            ("num",StandardScaler(),numerical_columns),
            ("ord",OrdinalEncoder(categories=[condition_order]),ordinal_columns),
            ("nom",OneHotEncoder(drop="if_binary"),nominal_columns)
        ])

    def create_model(self):
        print("\nCreating Model.....")
        self.model = Pipeline(
        steps=[
            ("preprocessor", self.preprocessor),
            ("regressor",LinearRegression())
        ])

    def train_model(self):
        print("\nStarting Model Training.....")
        print("\nData Types:")
        print(self.xtrain.dtypes)
        self.model.fit(self.xtrain,self.ytrain)
        print("\nModel Successfully Trained")

    def prediction(self):
        print("\nPredicting House Prices.....")
        self.pred = self.model.predict(self.xtest)
        print("\nPredicted Prices:")
        print(self.pred)
        print("\nActual Prices:")
        print(self.ytest.values)

    def evaluation(self):
        print("\nModel Evaluation")
        mae = mean_absolute_error(self.ytest,self.pred)
        mse = mean_squared_error(self.ytest,self.pred)
        rmse = mse ** 0.5
        r2 = r2_score(self.ytest,self.pred)
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("Root Mean Squared Error:", rmse)
        print("R2 Score:", r2)

    def save(self):
        print("\nSaving Model.....")
        joblib.dump(self.model,"Price Predictor Model.pkl")
        print("Model Successfully Saved")

    def predict_price(self):
        print("\nPredicting New House Price.....")
        new_house = pd.DataFrame([
        {
            "Area_SqFt": 700,
            "Bedrooms": 3,
            "Bathrooms": 4,
            "Floors": 2,
            "Year_Built": 2006,
            "Location": "Rural",
            "Condition": "Good",
            "Garage_Capacity": 1,
            "Has_Pool": "Yes",
            "Distance_City_Center_Miles": 10
        }
    ])
        prediction = self.model.predict(new_house)
        print("Price of This House:", prediction[0].round(3))

    def pipeline(self):
        self.load_data()
        self.cleaning_data()
        self.split_data()
        self.fe()
        self.create_model()
        self.train_model()
        self.prediction()
        self.evaluation()
        self.save()
        self.predict_price()

predictor = PricePrediction("house_prices_messy_dataset.csv")
predictor.pipeline()
