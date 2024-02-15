#!/usr/bin/env python
# coding: utf-8

# # Multivariate Exploratory Data Analysis: Geographic Choropleths
# ---
# ## Purpose:
# In the last notebook (Multivariate EDA Part 2) we explored the relationship between population-normalized pollution rates and population density (urbanization level) as defined by the USDA. While the last few notebooks answered the four key questions we set out to address in this project, <u>we will conclude the study with by exploring geographical  trends using choropleths.</u> Choropleths are thematic maps that represent statistical data. We can choropleths to demonstrate how pollution contributors will vary regionally (since prevalent industries and climate may vary), as well as variables like income, property values, and unemployment.   
# 
# *Note: Since our merged dataframe only has about half of U.S. Counties, we're going to focus analysis at the state level only*
# 
# 
# ### Package Installation and Versioning Requirments:
# For questions regarding python version, package installations, and other functional requirements, see the *Read Me* file contained [here](link).
# 
# Now, let's begin.

# ### Import Packages and Load Data

# In[102]:


# Importing required packages: 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import altair as alt
from urllib.request import urlopen
import json
import plotly.graph_objects as go
import plotly.express as px

# Importing py files
from Basic_data_structure_observations import *
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

# We'll add the normalized pollution column again for our geographical analysis  
df_merged_state["Emissions per Person"] = df_merged_state["EMISSIONS"] / df_merged_state["Civilian_labor_force_2021"] *2000


# Let's add a separate merged dataframe to consider emissions in sources where we don't have Redfin data 
merged_df = pd.merge(df_emissions_State, df_USDA_State, left_on='STATE', right_on="State", 
                     how='inner').reset_index()
merged_df["Emissions per Person"] = merged_df["EMISSIONS"] / merged_df["Civilian_labor_force_2021"] *2000


# ### Multivariate EDA: Choropleth Mapping

# In[156]:


def state_choropleth(df, column, states, title, normalize=True, _type = "Emissions", fig_num=10):
    """This function takes a dataframe, a feature column, and then uses the states specified to
    build a choropleth plot with the specified title. If the normalize input is set to true, it 
    will perform min-max normalization on the dataframe."""

    #Set dictionary to get Units, label
    _type_dict = {"EMISSIONS": ["TPY","Emissions"], 
                  "Emissions per Person": ["(lbs)", "CO per Capita"], 
                  "Civilian_labor_force_2021": ["Workers", "Labor Force Size"],
                  "Median_Household_Income_2021": ["(USD)","Median Income"], 
                  "Unemployment_rate_2021": ["(%)","Unemployment"],
                 "median_sale_price": ["(USD)","Property Value"]}
    
    # Error checking
    if column not in _type_dict:
        raise ValueError("Column '{}' not found in _type_dict".format(column))
    
    else: 
        _label = _type_dict[column][1]
        _units = _type_dict[column][0]

      
        
    # Find min/maxes:
    if normalize == True:
        _min = min(df[column])
        _max = max(df[column])
        
        # Min-max normalization through list comprehension
        values = [(val - _min)/(_max - _min) for val in df[column]]
    else:
        values = df[column]
        #Convert to tpy if State aggregated emissions: 
        if column == "EMISSIONS":
            values = values/2000       
            
    # Add text for callout label 
    text = ["{}: {:.0f} {}".format(state, value,_units) for state, value in zip(df[states], values)]
    #text = [f"{state}<br>Value: {value:.0f} lbs" 
    #        for state, value in zip(df[states], df[column])]   
        
    # Create figure
    fig = go.Figure(data=go.Choropleth(
    locations=df[states],
    z=values.round(3).astype(int),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=True,
    colorbar_title=_label+" " + _units,
    text=text 
    ))

    # Update figure layout for readability
    fig.update_layout(
      width=700,
      height=500,
      title_text = title,
      title_x=0.5,
      title_font_size=20,
      geo_scope='usa', # limit map to USA,
    )

    if column == "median_sale_price":
        # Add disclaimer that 4 states are not in merged dataframe
        fig.add_annotation(
            x=0.5,  
            y=0.95, 
            text="4 States Excluded: No data for MT, ND, SD, or WY",
            showarrow=False,  # Hide the arrow
            font=dict(size=12, color="red")  # Customize the font size and color
        )
    
    # Add caption: 
    fig.add_annotation(
        x=0.5,  
        y=-0.05,  
        text="Fig {} Choropleth of {}".format(fig_num, title),
        showarrow=False,  # Hide the arrow
        font=dict(size=12, color="black")  # Customize the font size and color
    )
    
    # Show plot:
    fig.show()
    return()


# In[157]:


state_plot = state_choropleth(merged_df, "Emissions per Person", "STATE",
                              "Emissions by State: Normalized by Population", False, fig_num=10) #lbs per person


# Interesting, so when we look at the population-normalized residential emissions across the US, we actually see that most states have CO emissions <200 lbs/per person each year.  However, we see some clear outliers. Alaska for example, has > 700lbs/per of CO emissions. If we recall from our last notebook, the correlation heat map suggested that population density was the strongest indicator of population-normalized pollution levels. 
# 
# And if we look geographically, we see more rural states like Wyoming, Montana, and Maine seem to end up on the higher end of per person pollution.
# 
# What about total emissions, income, unemployment rates, and property values? How are these distributed?
# let's look at those next. 

# In[158]:


# Examine total emissions: 
plot = state_choropleth(merged_df, "EMISSIONS", "STATE",
 "Total Emissions Across the Country", False, fig_num=11) #lbs per person


# Here, we clearly see why normalizing by population is important to consider total emissions. Alaska has the greatest individual pollution rate of any state for residential emissions, but on a total pollution basis it's contributions are a fraction of Texas's pollution which sits at >1MM Tons/year. 
# 
# Let's continue and see how population (approximated by labor force) is distributed across the US. 

# In[145]:


# Examine total emissions: 
plot = state_choropleth(merged_df, "Civilian_labor_force_2021", "STATE",
 "Labor Force Distributions Across the Country", False,fig_num=12) #lbs per person


# Great, we see that California has the greatest population, but you might recall their residential pollution levels are actually less than the state of North Carolina.

# In[146]:


# Examine total emissions: 
plot = state_choropleth(merged_df, "Median_Household_Income_2021", "STATE",
 "Income Distributions Across the Country", False,fig_num=13) #lbs per person


# Here we can clearly see there are pockets of high income areas such as the East and West Coasts.

# In[147]:


# Examine total emissions: 
plot = state_choropleth(df_merged_state, "median_sale_price", "STATE",
 "Home Prices Across the County", False, fig_num=14) #lbs per person


# 

# In[148]:


# Examine total emissions: 
plot = state_choropleth(merged_df, "Unemployment_rate_2021", "STATE",
 "Unemployment Across the County", False, fig_num=15) #lbs per person


# Finally, we can see the unemployment rate distribution across the US. Now this was measured during the pandemic, and so we shouldn't be surprised that some starts were particularly devastated by the pandemic shutdown. California for example, banned operating barbershops and similar businesses during this time, which likely lead to exacerbated unemployment rates. Meanwhile, places like UT which had minimal pandemic resitrictions, along with a strong and diversified economy held an exceptionally low unemployment rate.    

# 
# ### Project Summary: 
# Throughhout this project we applied various scriping methods to clean, join, slice, and aggregate the combined datasets from the EPA, USDA, and Redfin. We used plotting packages like matplotlib, plotly, and Seaborn to generate meaningful inferences about how trends were distributed across the US.
# 
# We found that:
# * sectoral emissions varied heavily based on relative wealth of an area and population density.
# * Higher income regions *generally* were associated with lower pollution rates. 
# * rural regions were associated with greater pollution rates
# * On a pollution/person basis, emission reduction credits could be more effictive if they prioritized rural demographics
# 
# There's a lot more we could study. For more detail on next steps, view our formal slidedoc report. 
# 
# 
# ### End of Notebook
# That's the end of this notebook, and our correlation project. 
# We hope you've enjoyed this project. Sincerely, <br>
# Matt Jones, Hector Estrada Tock <br>
# MADS | University of Michigan
# 
# 
# ---

# In[ ]:




