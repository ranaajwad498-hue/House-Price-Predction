import pandas as pd

class priceprediction:
    def __init__(self, dataset):
        self.dataset_file=dataset
        self.dataset=None

    def load_data(self):
        self.raw_data= pd.read_csv("house_prices_messy_dataset.csv")
        # print(self.raw_data.head())
        # print("Inforamtion of Data Set:",self.raw_data.info())
      

    def cleaneing_data(self):
        self.df = self.raw_data
        # print("Duplicate Records In Data sets....")
        self.df.drop_duplicates()
        self.df["Location"] = self.df["Location"].replace(["SUBURBS",'downtown'],['Suburbs' ,'Downtown'])
        self.df["Condition"]=self.df["Condition"].replace(['FAIR','good'],['Fair','Good'])
        self.df["Has_Pool"]= self.df["Has_Pool"].replace(['Y','N',"TRUE","FALSE"],["Yes","No","Yes","No"])
        print(self.df["Has_Pool"].unique())
        print("Filling Null Values....")
        float_col=self.df.select_dtypes(include="float64")
        for col in float_col:
            self.df[col]=self.df[col].fillna(self.df[col].median())
    
        str_col= self.df.select_dtypes(include="str")
        for col in str_col:
            self.df[col]=self.df[col].fillna(self.df[col].mode()[0])
        print("Total Null Values in Data Set.....",self.df.isnull().sum())
        print(self.df.dtypes)






        



predictor=priceprediction("house_prices_messy_dataset.csv")
predictor.load_data()
predictor.cleaneing_data()