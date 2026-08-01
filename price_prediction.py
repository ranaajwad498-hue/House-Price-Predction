import numpy as np
import pandas as pd

class priceprediction:
    def __init__(self, dataset):
        self.dataset_file=dataset
        self.dataset=None

    def load_data(self):
        self.raw_data= pd.read_csv("house_prices_messy_dataset.csv")
        print(self.raw_data.head())
        print("Inforamtion of Data Set:",self.raw_data.info())
      

    def cleaneing_data(self):
        self.df = self.raw_data
        # print("Duplicate Records In Data sets....")
        self.df.drop_duplicates()
        print("Filling Null Values....")
        float_col=self.df.select_dtypes(include="float64")
        for col in float_col:
            self.df[col]=self.df[col].fillna(self.df[col].median())
    
        str_col= self.df.select_dtypes(include="str")
        for col in str_col:
            self.df[col]=self.df[col].fillna(self.df[col].mode()[0])

        self.df["Location"] = self.df["Location"].replace(["SUBURBS",'downtown'],['Suburbs' ,'Downtown'])
        self.df["Condition"]=self.df["Condition"].replace(['FAIR','good'],['Fair','Good'])
        self.df["Has_Pool"]= self.df["Has_Pool"].replace(['Y','N',"TRUE","FALSE"],["Yes","No","Yes","No"])
        self.df["Bathrooms"]= self.df["Bathrooms"].replace([1.5, 2.5, 3.5,4.5],[2,3,4,5])
        self.df["Area_SqFt"] = self.df["Area_SqFt"].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        self.df["Area_SqFt"]=self.df["Area_SqFt"].fillna(self.df["Area_SqFt"].median())
        self.df["Bedrooms"] = self.df["Bedrooms"].clip(lower=0, upper=5)
        self.df["Price"] = self.df["Price"].clip(lower=0, upper=900000)
        print("Checking Null Values.....",self.df.isnull().sum())
        self.df.to_csv("Cleaned Data.csv",index=False)


        







        



predictor=priceprediction("house_prices_messy_dataset.csv")
predictor.load_data()
predictor.cleaneing_data()