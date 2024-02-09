#!/usr/bin/env python
# coding: utf-8

# # Basic Data Structure Observations
# ---
# ## Purpose:
# In this notebook we're going to do our first exploration of our EPA, USDA, and Redfin data. Eventually, we will combine these three datasets to identify correlations between pollution and various indicators of wealth. To start though, it's important to evaluate each dataset and the potential issues that could arise from using them. These issues could include:
# * **ethical concerns** <br> Such as considering how the data was collected, if/how consent was obtained, how the data will be used, and how it could cause harm or shift the balance of power.  <br><br>
# * **technical concerns**  <br> such as considering the magnitude of the data, it's structure, data reliability, and how missing data should be treated.     
# 
# As we explore the data, a key ethical concern could involve how the findings would be use to enact legislative change, or adjust public opinion about particular groups. As such, <u>we strictly prohibit the aggregation of senstive population data</u> (race, religion, age, etc) with this study. Likewise this study should not be used to endorse proposed pollution legislation, without expansive supporting evidence.  A detailed *Terms of Use* clause is maintained [here](link) regarding how this work may be cited and/or used. Before using this research, you must agree to adhere to the Terms of Use and copyright limitations.  
# 
# ### Package Installation and Versioning Requirments:
# For questions regarding python version, package installations, and other functional requirements, see the *Read Me* file contained [here](link).
# 
# Now, let's review the data structure we have.

# In[25]:


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

#Get the data
df_emissions, df_USDA, df_Redfin = load_base_data()

#highlight missing data if any
print(check_nan(df_emissions))

#view sample -- starting with emission data
df_emissions.info()


# In[20]:


#Let's also take a sample to see the data contents
df_emissions.sample(4)


# In[33]:


# Okay, so we see 9 columns, including a unique ID (FIPS) for the state and counties in the emissions dataset. 
# Emissions are separated by sector, which could be interesting to probe differences in pollution contributors,
# and we have the county name and state abbreviation for each emission source. Now since we extracted the data
# from EPA's NEI, we know that the only pollutant considered is carbon monoxide, so we can ignore those columns
# in the future. 

# Now, the dataframe has no missing values, but that doesn't mean the dataframe contains all 
# counties in the US. We'll check on that for all datasets shortly. Also, while it's not explicitely stated, 
# this is 2020 emission data only. This will be a problem for our USDA data and you'll see why shortly.

# Let's explore that data now. 


# In[34]:


#checking the USDA data: 
print(check_nan(df_USDA))

#view sample -- starting with emission data
df_USDA.info()


# In[27]:


df_USDA.sample(3)


# In[31]:


# So there's a lot to unpack here. First -- we have lots of columns, based on the variable of interest, and the 
# year of the data. Now there are some really awesome features in this dataset including income, 
# unemployment rates, labor force size, location, and rural/urban designation code (Continuum Code). 

# After a few minutes some issues should be apparent. First, we don't have income data for 2020, and the last rural
# continuum code record was taken in 2013. These variables could be really interesting to probe in our analysis. 

# Since we could not find readily available data to compensate here, we plan to use both of these variables, but
# there's some pretty big assumptions here: 
# 1. We assume income changes between 2020 and 2021 are neglgible down to the county level. For some counties this 
#    could be vastly different from reality given the COVID pandemic's impact on local economies. 
# 2. We assume the rural urban continuum is reasonably close to what it was in 2013. Again this could be an issue 
#    as some regions substantial growth during this period, like Austin TX while other areas had swaths of emigration
#    (like Edenville, MI where a dam break forces residents to leave)


# In[32]:


#Before we dive further into this, let's review the Redfin dataset

#checking the USDA data: 
print(check_nan(df_Redfin))

#view sample -- starting with emission data
df_Redfin.info()


# In[29]:


df_Redfin.sample(3)


# Okay, so at a high-level we see that the Redfin dataset provides information on sale price, property type, 
# location, number of listings, and the period when the sale occured. Notably, the state name is not 
# abbreviated and we aren't given any FIPs to connect to the other datasets, so that's going to be 
# an issue down the road. We also only want 2021 data (since our income data is from the same time period, 
# and we assume emissions are the same in 2020 and 2021). 
# 
# Now that we've scratched the surface, we'd like to see if we can make more sense of the data graphically, so we can understand how factors like income, pollution, unemployment, and sale prices are distributed. 
# 
# To do this graphically, we need to some light data cleaning and grouping, which is covered in the next notebook. 

# ### End of Notebook

# Next notebook: Data_manipulation
# 
# 
# *Note: to limit the number of functions duplicated, all codebook functions will be saved in py files that can be imported to execute.*
# 
# ---
# 

# In[ ]:




