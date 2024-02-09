#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning and Manipulation
# ---
# ## Purpose:
# Now that we have a cursory level of understanding of the data, we want to graphically understand how the data is distributed. To do that we need to establish some functions to clean and organize the data. This codebook contains functions to clean the data and organize it, so we can conduct meaningful exploratory data analysis (EDA). 
# 
# ### Package Installation and Versioning Requirments:
# For questions regarding python version, package installations, and other functional requirements, see the *Read Me* file contained [here](link).
# 
# Now, let's begin.

# ### Import Packages and Load Data

# In[1]:


import pandas as pd
import Basic_data_structure
import numpy as np
df_emissions, df_USDA, df_Redfin = Basic_data_structure.load_base_data()


# ### Basic Data Cleaning Concepts

# In[2]:


# There's a couple things we want to consider with the data cleaning. Ultimately we want: 
#    1. Flexibility so we can use the function(s) for univariate and multivariate analysis
#    2. Capable of cleaning for merged and non-merged data
#    3. Efficient considering the significant record counts.

# To do this, we're going to make a few functions that allow us to group the data by county and state levels
# and we're also going to allow for aggregation so we can see total emissions by region, or sector specific based on
# user input. 


# let's go ahead and do this. To start we need to have some weighted avg. functions to enable us to capture a closer 
# estimate of income. 
def weighted_avg(df,series):
    """Generic function for calculating weighted averages based on the population. This function works for groupby
    applications, where the variable you want to take the wegithed average of is listed in the aggregation 
    dictionary, where you would put the weighted_avg function in the value segment. The labor force column must 
    exist in the dataframe."""
    
    return np.average(series, weights=df.loc[series.index, 'Civilian_labor_force_2021'])



# ### Redfin Data Cleaning

# In[3]:


df_Redfin.head(2)


# In[4]:


# We'll start with our Redfin data. Since we only want 2021 data, we're going to apply transformations to 
# convert the period data to just 2021, so we'll slice just to capture Starting on Jan 1st of that year, and
# ending by Dec. 31st. Now it could be interesting to explore real estate prices and fluctuations in other years
# but that's out of scope for this project, we just want to understand houses prices in 2021 as they relate to 
# expected emissions. 


def Clean_Redfin(df, level="STATE"):
    """this function cleans the Redfin dataset for our analysis. While there are many factors in the
    dataset, we're just going to load those we're interested in for Wealth vs Pollution analysis. This
    function requires the Redfin dataframe input and a specification of the level of aggregation 
    (STATE or COUNTY)"""

    #Convert period data to timeseries object so we can pull from 2021 only. 
    df[['period_begin', 'period_end']] = df[['period_begin', 'period_end']].apply(pd.to_datetime)
    
    #Filter for sales data from 2021. We're also only going to consider residential properties. 
    df = df[(df["period_begin"].dt.year==2021) & (df["property_type"]=="All Residential") & 
            (df["region_type"]=="county")].reset_index() 
    
    #Remove nan sale entries: 
    df = df.dropna(subset="median_sale_price")
    
    if level == "STATE":
        #Take median sale prices at the state level
        df = df.groupby("state").agg(
            {'median_sale_price': "median","state_code": "first"}).reset_index() 
        
        #also let's exclude DC. It's not a state.
        df = df[df["state"] != "Columbia"]        

    elif level == "COUNTY":
        #Take median sale prices at the county level
        df = df.groupby("region").agg(
            {'median_sale_price': "median", "state": "first", "state_code": "first"}).reset_index() 
    else: 
        raise ValueError("Level must be entered as 'STATE' or 'COUNTY'.")
        
    # Next let's round the sale prices. Generally the prices have 5 significant digits, so we'll reduce broadly
    # We're taking the log of the sale price (abs in case quit-claims), rounding to the nearest integer using floor
    # and converting the float to an int. 5 is our sig digits, so N-int(...) gets use num digits to round by. 
    # then as a lambda function, we're rounding each element (x = row entry) by N-int digits of the sales column
    N=5
    df['median_sale_price'] = df['median_sale_price'].apply(lambda x: round(x, N - int(np.floor(np.log10(abs(x))))))

    return(df)

df_Redfin_Clean = Clean_Redfin(df_Redfin, "COUNTY")
df_Redfin_Clean.describe()


# In[5]:


# So we see we have 46 states with sale price data, not too bad. And we can see the home prices skew
# higher by about 20-25k. Not too bad. Let's now clean the USDA dataset. 


# ### USDA Data Cleaning

# In[6]:


df_USDA.head(2)


# In[7]:


def Clean_USDA(df, level="STATE"):
    """This function cleans the USDA dataset and performs aggregation  based on the level (COUNTY or STATE)
    based on user declarations."""
    
    # First, let's down-select to the columns we're interested in. Recall from the last notebook, that we 
    # will keep the 2013 rural/urban rankings as a coarse way to evaluate emission differences based on
    # population density. 
    df = df[["State", "Area_Name", 'Civilian_labor_force_2021', "Rural_Urban_Continuum_Code_2013",
             "Median_Household_Income_2021", "Unemployment_rate_2021", "FIPS_Code",]].astype({"FIPS_Code": str})
    
    #remove areas with nan for income, unemployment, or labor force: 
    df = df.dropna(subset =["Unemployment_rate_2021", "Median_Household_Income_2021", "Civilian_labor_force_2021"])
    
    # We want to set the FIPS code to string type. The reason will be apparent when we merge this
    # data with our emission data. The zfill, ensures that each string is left padded with zeros to make each
    # 5 character length. In case a code (003 becomes clipped as 3)
    df["FIPS_Code"] = df["FIPS_Code"].astype(str).str.zfill(5)
     
    # Filter to the columns of interest. From this dataset we want the income, unemployment, labor force size, 
    # rural/urban designation, and the FIPs. Brining in Area_name is not necessary, but helpful for reading
    # in our initial exploration
    cols = ["FIPS_Code", "Area_Name", "State", "Civilian_labor_force_2021", 
            "Median_Household_Income_2021", "Unemployment_rate_2021", "Rural_Urban_Continuum_Code_2013"]
    df = df[cols]
    
    # U.S. Territories are out of scope for this project, so we're removing them. The ~ is a negation operator
    # basically saying give me all the records where the states are not in the exclusion list. 
    exclusion_list = ["PR", "DC", "US"]
    df = df[~df["State"].isin(exclusion_list)].reset_index(drop=True)
    
    #We'll add a column for the state fips which will be usefull later
    df["State_FIPS"] = df['FIPS_Code'].apply(lambda x: x[:2])
    
    # similar to the Redfin data, we're going to group this by level specification. Since we want to weight the 
    if level=="STATE":
        agg_dict = { "FIPS_Code": "first", "Civilian_labor_force_2021": "sum", 
                    "Median_Household_Income_2021": lambda x: weighted_avg(df, x),
                    "Unemployment_rate_2021":lambda x: weighted_avg(df, x), "State_FIPS": "first"}

        df = df.groupby("State").agg(agg_dict).reset_index()
        
        return(df)
    else: 
        return(df)
df_USDA_Clean = Clean_USDA(df_USDA, "COUNTY")
df_USDA_Clean.describe()


# In[8]:


# Okay, we see that we have all 50 states accounted for in our dataset, and the numbers look reasonable. 
# The Median income, averaged across all 50 states is roughly 70k, which is pretty close to estimates from
# Census.Gov for 2021. Labor force size is going to be right skewed (at state level, and Unemployment seems
# to fall in line with the median. When we plot this we'll understand it better). 

#Let's now clean the Emission Data. This is going to be much more involved than the other datasources. 


# ### EPA Data Cleaning 

# In[9]:


# Since the EPA dataset has emissions by sectors, and we might want to probe sectoral as well as regional differences
# We're going to first create some background functions to help us deal with condensing the sectors into primary groups 
# first going to create a function to allow # us to condense the dataframe to Major sectors like (Mobile, Industrial,
# fuel combustion, etc.)
df_emissions.head(2)


# In[10]:


def group_emissions_by_major_sectors(df, level, options_dict):
    """This function takes in the emissions dataframe, a specified grouping level, an input dictionary, which 
    will contain keys that are the major sectors we'd like to consider and a list of string values for each 
    major sector key. """
    
    df_list = []
    # We're going to do this by calling the K-V pairs using dict.items(). sector is our key, and options is our 
    # value list. We'll be iterating using a for loop. 
    
    for sector, options in options_dict.items():
        #make a subset dataframe where the sectors are only those specified in the current K-V pair. 
        filtered_df = df[df['SECTOR'].isin(options)]
        
        # add the dataframe to a list, while adding a new column for our Primary sectors (the keys in our options 
        # dictionary)
        df_list.append(filtered_df.assign(Major_Sector=sector))
    
    # Once done looping, concat the dataframes. 
    agg_df = pd.concat(df_list)
    
    # Next we'll use groupby method to group the emissions based on the major sector and the Count/State FIPS 
    # Depending on the level specified. 
    if level == "COUNTY": 
        grouped_df = agg_df.groupby(["COUNTY FIPS", "Major_Sector"]).agg(
            {'EMISSIONS': "sum", "STATE": "first", "STATE FIPS": "first", "COUNTY": "first"}).reset_index()
    elif level == "STATE":
        grouped_df = agg_df.groupby(["STATE FIPS", "Major_Sector"]).agg(
                    {'EMISSIONS': "sum", "STATE": "first"}).reset_index()
    else: 
        raise ValueError("You must specify 'COUNTY' or 'STATE' for the level.")
    return grouped_df


# In[1]:


#Next let's make our function to supply the options_dict to our sector-grouping function above.
def create_options_dict(input_set):
    """This function creates the options_dict that will be supplied when group_emissions_by_major_sector
    is called. This function required the input_dict of options, which will be the set of all sectors provided. """

    # We want just 6 main categories, but there's not a broadly applicable way to directly do this. So, we're 
    # going to map those connections to the primary sector here.  
    category_mapping = {
        'Solvent': 'Industrial Processes',
        'Gas Stations': 'Commercial',
        'Agriculture': 'Industrial Processes',
        'Commercial Cooking': 'Commercial',
        'Bulk Gasoline Terminals': 'Industrial Processes',
        'Waste Disposal': 'Industrial Processes',
        'Dust': 'Industrial Processes',
        'Biogenics': "Miscellaneous Non-Industrial NEC"
    }

    # Now we're going to create our new dictionary containing our primary sectors, and the secondary level(s) as a 
    # value list of string entries. 
    options_dict = {}
    for option in input_set:
        
        # Identify Primary Sectors using the split method on ' - ', taking the first item from the resulting split list 
        category = option.split(' - ')[0]
        
        # use .get method to see if our primary sector is in our category_mapping dictionary. use category 2x to check
        # both the keys and the values
        mapped_category = category_mapping.get(category, category)
        
        #if the primary sector is not already in our new dictionary, we're going to add it. 
        if mapped_category not in options_dict:
            options_dict[mapped_category] = []
        
        # And for each element in the input_set we're going to add it as a value corresponding to the appropriate 
        # primary sectory key (mapped category)
        options_dict[mapped_category].append(option)
    options_dict
    return(options_dict)



# In[12]:


# And now let's setup our cleaning function so that depending on the level selected, sector desired, and emission 
# contributions, we are able to get the output dataframe we want. 


# In[13]:


def Clean_EPA(df, level="STATE", emission_contributor="residential", agg=True):
    """This function cleans the EPA Emission dataset and performs aggregation  based on the level (county or state), 
    sector inclusion/exclusion, and emission contributors (residential, industrial, etc). At the end, this function
    returns a dataframe ready for user analysis. Contributor options are 'residential', 'industrial', 'commercial',
    'by sector', or 'all'. """
    
    #Add a column for unity County, State FIPS ID: 
    if level=="COUNTY":
        df["FIPS"] = df["STATE FIPS"] + df["COUNTY FIPS"]

    
    #We will Filter based on emission_contributor: 
    emission_sectors = set(df_emissions["SECTOR"])
    
    if emission_contributor == "by sector": 
        #get the options dictionary 
        options_dict = create_options_dict(emission_sectors)
        #run the grouping function for the sectors
        df = group_emissions_by_major_sectors(df, level, options_dict)
    
    elif emission_contributor in ["residential", "industrial", "commercial"]:
        #Get dictionary of strings to search for each string entry in our sector list. 
        options_dict = {
            "residential": ["Residential", "Light Duty"],
            "commercial": ["Comm/Institutional", "Commercial", "Dry Cleaning", "Graphic Arts", "Gas Stations"],
            "industrial": ["Industrial", "Bulk", "Agriculture", "Waste", "Degreasing", "Electric Generation", 
                           "Locomotive", "Aircraft"]}
        
        # Create a list of desired sectors via list comprehension. We do this by iterating through each entry in 
        # emission_sectors, and checking if any of the string entries from the value list corresponding to our 
        # sector key are in the list by using the .get method on our options dict.  
        desired_sectors = [val for val in emission_sectors if any(option in val for option in options_dict.get(
            emission_contributor, []))]
        #reduce the dataframe to only consider those particular emissions contributors: 
        df = df[df["SECTOR"].isin(desired_sectors)]
        
    elif emission_contributor != "all":
        raise ValueError("""You must specify one of the following options for sector:\
        'residential', 'industrial', 'commercial','by sector', or 'all' """)
    
    # Now group based on level and aggregation specified. 
    df = df.copy(deep=True)
    
    if level == "STATE" and agg == True: 
        df = df.groupby("STATE FIPS").agg(
        {'EMISSIONS': "sum","STATE": "first"}).reset_index() 
            
    elif level == "COUNTY" and agg == True: 
        df = df.groupby("FIPS").agg(
        {'EMISSIONS': "sum","STATE FIPS": "first", "STATE": "first", 
         "COUNTY": "first", "COUNTY FIPS": "first"}).reset_index()
        
    elif level == "STATE" and agg == False: 
        try: 
            df = df.groupby(["Major_Sector", "STATE FIPS"]).agg(
                {'EMISSIONS': "sum", "STATE": "first"}).reset_index()         
        except: 
            df = df.groupby(["SECTOR", "STATE FIPS"]).agg(
                {'EMISSIONS': "sum", "STATE": "first"}).reset_index()        

    #We're also going to create a FIPS column containing the unique code associated with each county. 
    output_df = df.copy(deep=True)

    #if level is county, we'll add in the FIPS code: 
    if level == "COUNTY": 
        output_df["FIPS"] = df['STATE FIPS'].astype(str).str.zfill(2) + df['COUNTY FIPS'].astype(str).str.zfill(3)
        output_df["Region"] = output_df["COUNTY"] + ' County, ' + output_df["STATE"]
    #Finally, let's remove non-states
    exclusion_list = ["TR", "DM", "PR", "VI", "DC"]
    output_df = output_df[~output_df["STATE"].isin(exclusion_list)]
    
    #and return the dataframe: 
    return(output_df)


# In[14]:


df_emissions_Clean = Clean_EPA(df_emissions, "COUNTY", "residential", agg=True)
df_emissions_Clean.describe()


# In[15]:


# Now we have established functions for cleaning our EPA, Redfin, and USDA datasets, and preparing them for univariate
# analysis. We'll go ahead and finish our data cleaning and manipulation work by establishing a function for merging
# the data. 


# ### Merging the Cleaned data together: 

# In[24]:


def get_merge_df (df_emission, df_USDA, df_Redfin, level):
    """This function will merge the data together into a singular dataframe based on the loaded dataframes from
    the EPA, USDA, and Redfin, as well as the level at which merging occurs. Level specification is important
    since it influences how we will merge the data."""
    
    
    if level == "STATE": 
        #Provide list of merged columns we'll want
        keep_cols = ["STATE", "STATE FIPS", "EMISSIONS", "Civilian_labor_force_2021", 
                         "Median_Household_Income_2021", "Unemployment_rate_2021", "median_sale_price",
                        "state"]
        if "Major_Sector" in list(df_emission.columns): 
            keep_cols.append("Major_Sector")

        merged_df = pd.merge(df_emission, df_USDA, left_on='STATE FIPS', right_on="State_FIPS", 
                             how='inner').reset_index()
        merged_df = pd.merge(merged_df, df_Redfin, left_on='State', right_on="state_code", how='inner').reset_index()
        merged_df = merged_df[keep_cols]

    elif level == "COUNTY":
        #Provide list of merged columns we'll want
        keep_cols = ["STATE", "STATE FIPS", "COUNTY FIPS", "COUNTY", "Rural_Urban_Continuum_Code_2013", 
                     "EMISSIONS", "Civilian_labor_force_2021","Median_Household_Income_2021", 
                     "Unemployment_rate_2021", "median_sale_price"]
        if "Major_Sector" in df_emission.columns: 
            keep_cols.append("Major_Sector")   

        merged_df = pd.merge(df_emission, df_USDA, left_on='FIPS', right_on="FIPS_Code", 
                             how='inner').reset_index()
        merged_df = pd.merge(merged_df, df_Redfin, left_on='Region', right_on="region", how='inner').reset_index()
        merged_df = merged_df[keep_cols]        

    return(merged_df)


# In[23]:


#In the future we'll keep the code block below for simple calls to get our County and State dataframes for analysis.  
df_Redfin_County = Clean_Redfin(df_Redfin, "COUNTY")
df_USDA_County = Clean_USDA(df_USDA, "COUNTY")
df_emissions_County = Clean_EPA(df_emissions, "COUNTY", "by sector", agg=False)


df_Redfin_State = Clean_Redfin(df_Redfin, "STATE")
df_USDA_State = Clean_USDA(df_USDA, "STATE")
df_emissions_State = Clean_EPA(df_emissions, "STATE", "by sector", agg=False)


df_merged_state = get_merge_df(df_emissions_State,df_USDA_State,df_Redfin_State, level="STATE")
df_merged_cty = get_merge_df(df_emissions_County,df_USDA_County,df_Redfin_County, level="COUNTY")


# In[18]:


# Excellent, now we have functions that enable us to clean and filter the data with a lot of flexibility. If we want to 
# check pollution vs. wealth differences at state / county levels we can, and we can even drill down to whether we 
# want to include sectoral differences. 

# Obviously this flexibility comes at the cost of making a function which is less readible, and harder to immediately 
# understand. However, we chose this to maximize flexibility for future analysis so that only minor data manipulation 
# is needed down the road.


# ### End of Notebook
# 
# Next notebook: Univariate EDA 
# 
# ---

# In[21]:


df_merged_cty


# In[ ]:




