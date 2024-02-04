#import necessary packages: 
import pandas as pd


#We'll create function to load the data
def load_base_data(): 
    """This function will load the raw EPA, USDA, and Redfin datasets and return them as a list. 
    """
    df_emissions = pd.read_json("data/nei.json",dtype={'COUNTY FIPS': str, "STATE FIPS": str})
    df_USDA = pd.read_excel("data/Unemployment.xlsx",header=4)
    df_Redfin = pd.read_csv("data/county_market_tracker.tsv000", sep = '\t')
    return(df_emissions, df_USDA, df_Redfin)


#and let's add a function to count missing data in the datasets (if any)
def check_nan(df): 
    """Checking for null values"""
    return(df.isnull().sum())


