#!/usr/bin/env python
# coding: utf-8

# # Multivariate Exploratory Data Analysis: Geographic Choropleths
# ---
# ## Purpose:
# In the last notebook (Multivariate EDA Part 2) we explored the relationship between population-normalized pollution rates and population density (urbanization level) as defined by the USDA. While the last few notebooks answered the four key questions we set out to address in this project, <u>we will conclude the study with a bonus by exploring geographical  trends using choropleths.</u> Choropleths are thematic maps that represent statistical data. We can choropleths to demonstrate how pollution contributors will vary regionally (since prevalent industries and climate may vary), as well as variables like income, property values, and unemployment.   
# 
# 
# ### Package Installation and Versioning Requirments:
# For questions regarding python version, package installations, and other functional requirements, see the *Read Me* file contained [here](link).
# 
# Now, let's begin.

# ### Import Packages and Load Data

# In[51]:


#importing required packages: 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import altair as alt
from urllib.request import urlopen
import json
import plotly.graph_objects as go
import plotly.express as px

# Importing py files
from Basic_data_structure import *
from Data_manipulation import *
from Univariate_EDA import get_extremes
# Load the dataframes
global df_emissions, df_USDA, df_Redfin
df_emissions, df_USDA, df_Redfin = load_base_data()

# First we're going to load in the data. Now, since we want to view the geographical distribution of emissions, 
# we're going to set the aggregate to True, so we have just one value for each region plotted (county/state). 
# For this study, we're going to focus on the state level only since we only have 1/2 of the US counties in 
# our dataset. 


# Clean the county-level dataframes, focusing on residential emissions
df_Redfin_County = Clean_Redfin(df_Redfin, "COUNTY")
df_USDA_County = Clean_USDA(df_USDA, "COUNTY")
df_emissions_County = Clean_EPA(df_emissions, "COUNTY", "residential", agg=True)

# Clean the state-level dataframes, focusing on residential emissions
df_Redfin_State = Clean_Redfin(df_Redfin, "STATE")
df_USDA_State = Clean_USDA(df_USDA, "STATE")
df_emissions_State = Clean_EPA(df_emissions, "STATE", "residential", agg=True)

#Get Merged Dataframes to work with 
df_merged_state = get_merge_df(df_emissions_State,df_USDA_State,df_Redfin_State, level="STATE")
df_merged_cty = get_merge_df(df_emissions_County,df_USDA_County,df_Redfin_County, level="COUNTY")

# We'll add the normalized pollution column again for our geographical analysis  
df_merged_state["Emissions per Person"] = df_merged_state["EMISSIONS"] / df_merged_state["Civilian_labor_force_2021"] *2000
df_merged_cty["Emissions per Person"] = df_merged_cty["EMISSIONS"] / df_merged_cty["Civilian_labor_force_2021"] *2000
df_merged_cty["FIPS"] = df_merged_cty['STATE FIPS'].astype(str).str.zfill(2) + df_merged_cty[
    'COUNTY FIPS'].astype(str).str.zfill(3)


# ### Multivariate EDA: Choropleth Mapping

# In[52]:


def county_choropleth(df, column, FIPS, title, normalize=True):
    """This function takes a dataframe, a feature column, and then uses the FIPS specified to build a choropleth plot at the county level.
    The user can specify if a single state or if multiple states are included, and dynamically correct the plotting geometry,. If the normalize
    input is set to true, it will perform min-max normalization on the dataframe"""

    #Read in list of US counties from plotly based on FIPS codes
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    _min = min(df[column])
    _max = max(df[column])
    
    #prepare max-min normalization
    if normalize == True:
        #min-max normalization through list comprehension
        values = [(val - _min)/(_max - _min) for val in df[column]]
        
    else:
        values = df[column]

    fig = px.choropleth(df, geojson=counties, locations=FIPS, color=column,
                            color_continuous_scale="Viridis",
                            range_color=(_min, _max),
                            scope="usa",
                            labels={'unemp':'unemployment rate'}
                            )
    
    #fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    fig.update_layout(
      width=700,
      height=500,
      title_text = title,
      #colorbar_title=str.lower(column).capitalize(),
      title_x=0.5,
      geo_scope='usa', # limit map to USA,
    )
    
    fig.show()
    return()


def state_choropleth(df, column, states, title, normalize=True):
    """This function takes a dataframe, a feature column, and then uses the states specified to build a choropleth plot with the specified title.
    If the normalize input is set to true, it will perform min-max normalization on the dataframe"""

    #find min/maxes:
    if normalize == True:
        _min = min(df[column])
        _max = max(df[column])
        #min-max normalization through list comprehension
        values = [(val - _min)/(_max - _min) for val in df[column]]
    else:
        values = df[column]

    #Create figure
    fig = go.Figure(data=go.Choropleth(
    locations=df[states],
    z=values,#.round(-3).astype(int),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=True,
    #marker_line_color=color_list,
    colorbar_title=str.lower(column).capitalize()
    ))

    #Update figure layout for readability
    fig.update_layout(
      width=700,
      height=500,
      title_text = title,
      title_x=0.5,
      geo_scope='usa', # limit map to USA,
    )

    #show plot:
    fig.show()
    return()


# In[56]:


cty_plot = county_choropleth(df_merged_cty, "Emissions per Person", "FIPS", "County-Level Emissions", False,) #lbs per person
state_plot = state_choropleth(df_merged_state, "Emissions per Person", "STATE",
 "Emissions by State: Normalized by Population", False) #lbs per person


# Interesting, so when we look at the population-normalized residential emissions across the US, we actually see that most states have CO emissions that range from 100-200 lbs/per person each year.  However, we see some clear outliers. Alaska for example, has > 700lbs/per of CO emissions. If we recall from our last notebook, the correlation heat map suggested that population density was the strongest indicator of population-normalized pollution levels. 
# 
# And if we look geographically, we see more rural states like Main, Vermont, and New Mexico seem to end up on the higher end of per person pollution.
# 
# What about total emissions, income, unemployment rates, and property values? How are these distributed?
# let's look at those next. 

# In[28]:


# Examine total emissions: 
plot = state_choropleth(df_merged_state, "EMISSIONS", "STATE",
 "Emissions by State: Normalized by Population", False) #lbs per person


# Here, we clearly see why normalizing by population is important to consider total emissions. Alaska has the greatest individual pollution rate of any state for residential emissions, but on a total pollution basis it's contributions are a fraction of Texas's pollution which sits at >1MM Tons/year. 
# 
# Let's continue and see how population (approximated by labor force) is distributed across the US. 

# In[34]:


# Examine total emissions: 
plot = state_choropleth(df_merged_state, "Civilian_labor_force_2021", "STATE",
 "Emissions by State: Normalized by Population", False) #lbs per person


# Great, we see that California has the greatest population, but you might recall their residential pollution levels are actually less than the state of North Carolina.

# In[31]:


# Examine total emissions: 
plot = state_choropleth(df_merged_state, "Median_Household_Income_2021", "STATE",
 "Emissions by State: Normalized by Population", False) #lbs per person


# Here we can clearly see there are pockets of high income areas such as the north east around NY, NJ, and CT, as well as the West with CA, OR, and UT. 

# In[30]:


# Examine total emissions: 
plot = state_choropleth(df_merged_state, "median_sale_price", "STATE",
 "Emissions by State: Normalized by Population", False) #lbs per person


# 

# In[33]:


# Examine total emissions: 
plot = state_choropleth(df_merged_state, "Unemployment_rate_2021", "STATE",
 "Emissions by State: Normalized by Population", False) #lbs per person


# Finally, we can see the unemployment rate distribution across the US. Now this was measured during the pandemic, and so we shouldn't be surprised that some starts were particularly devastated by the pandemic shutdown. California for example, banned operating barbershops and similar businesses during this time, which likely lead to exacerbated unemployment rates. Meanwhile, places like UT which had minimal pandemic resitrictions, along with a strong and diversified economy held an exceptionally low unemployment rate.    

# ### End of Notebook
# 
# That's the end of this notebook, in the next notebook we'll conclude with a few remarks. 
# 
# Next notebook: Multivariate Analysis: Geographic Choropleths
# 
# ---

# In[ ]:


df_Redfin_State = Clean_Redfin(df_Redfin, "STATE")
df_USDA_State = Clean_USDA(df_USDA, "STATE")
df_emissions_State = Clean_EPA(df_emissions, "STATE", "residential", agg=True)

# Merge to our state level dataset  
df_merged_state = get_merge_df(df_emissions_State,df_USDA_State,df_Redfin_State, level="STATE")


# In[42]:


len(set(df_emissions_State["STATE"])), len (df_Redfin_State["state"]), len (df_USDA_State["State"])


# In[ ]:




