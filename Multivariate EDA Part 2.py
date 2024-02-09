#!/usr/bin/env python
# coding: utf-8

# # Multivariate Exploratory Data Analysis: Sectoral Analysis
# ---
# ## Purpose:
# In the last notebook (Multivariate EDA Part 1) we explored the relationship between income and property values. By using combo charts of bar and line graphs we found that the highest population-normalized pollution rates are found in areas with low population density, while the lowest rates seemed to skew towards fairly large and expensive metropolises like San Francisco. Then, by using heatmaps we observed that population density correlated better with residential pollution than income or property value. 
# 
# In this notebook, we're going to take the analysis a bit further. Since we have the option to load emission data at the sectoral level, <u>we will examine how population-normalized pollution rates might have different sectoral distributions across income bands or population densities.</u>         
# 
# We also want to see how each of our variables are distributed across the US. So after analyzing the sectoral emissions, we will develop a few choropleths to examine potential regional trends. 
# 
# ### Package Installation and Versioning Requirments:
# For questions regarding python version, package installations, and other functional requirements, see the *Read Me* file contained [here](link).
# 
# Now, let's begin.

# ### Import Packages and Load Data

# In[8]:


#importing required packages: 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import altair as alt

# Importing py files
from Basic_data_structure import *
from Data_manipulation import *
from Univariate_EDA import get_extremes
# Load the dataframes
global df_emissions, df_USDA, df_Redfin
df_emissions, df_USDA, df_Redfin = load_base_data()

# First we're going to load in the data. Now, since we want to examine sectoral emissions, we're going to turn off 
# the aggregation feature in our code. (agg = False), and we're going to specify "by sector" instead of residential. 
# There's more than 20 sectoral categories, so this helps us condense it to a managable list of just X instead. 

# Clean the county-level dataframes, focusing on residential emissions
df_Redfin_County = Clean_Redfin(df_Redfin, "COUNTY")
df_USDA_County = Clean_USDA(df_USDA, "COUNTY")
df_emissions_County = Clean_EPA(df_emissions, "COUNTY", "by sector", agg=False)

# Clean the state-level dataframes, focusing on residential emissions
df_Redfin_State = Clean_Redfin(df_Redfin, "STATE")
df_USDA_State = Clean_USDA(df_USDA, "STATE")
df_emissions_State = Clean_EPA(df_emissions, "STATE", "by sector", agg=False)

#Get Merged Dataframes to work with 
df_merged_state = get_merge_df(df_emissions_State,df_USDA_State,df_Redfin_State, level="STATE")
df_merged_cty = get_merge_df(df_emissions_County,df_USDA_County,df_Redfin_County, level="COUNTY")

# Normalize pollution at the person level, and convert from tons to lbs of CO emissions:
df_merged_state["Emissions per Person"] = df_merged_state["EMISSIONS"] / df_merged_state["Civilian_labor_force_2021"] *2000
df_merged_cty["Emissions per Person"]   = df_merged_cty["EMISSIONS"]   / df_merged_cty["Civilian_labor_force_2021"] *2000


# ### Multivariate EDA: Sectoral Analysis

# In[9]:


# First we're going to use list comprehension to establish a list of unique dataframes for each regional_sectoral
dfs_sectoral = [df_merged_cty[df_merged_cty['Rural_Urban_Continuum_Code_2013'] == code] for code \
                in set(df_merged_cty["Rural_Urban_Continuum_Code_2013"])]


# In[10]:


# Next we're going to compute the weighted average of the pollution ratios for each sector. We're going to use this as 
# our baseline, and then compare everything against that. 
def get_sectoral_baseline(df):
    """This function calculates the baseline composition of sectoral emissions by getting a unique set of primary 
    sectors, and calculating the weighted avg emissions from each sector. Then given the weighted average, computing 
    the relative composition as a percentage."""
    
    sectors = set(df["Major_Sector"])
    sectoral_wt_avg = {}
    for sector in sectors:
        temp_df = df[df["Major_Sector"]==sector]
        mean_val = weighted_avg(temp_df, temp_df["Emissions per Person"])
        sectoral_wt_avg[sector] = mean_val

    total_emissions = sum(sectoral_wt_avg.values())
    composition = {sector: (wt_avg / total_emissions) * 100 for sector, wt_avg in sectoral_wt_avg.items()}
    return(composition)

sectoral_baseline = get_sectoral_baseline(df_merged_cty)
sectoral_baseline


# In[11]:


# Next we're going to compute the baseline for each dataframe in our list that has been stratified based on 
# population density (Rural urban continuum code)

# Then, to accentuate the differences between each region and their emission distributions, we're going to 
# calculate the nominal drift that each location has from the baseline. So for example, if the avg composition 
# of fire emissions is 58%, but there are no emissions in a particular county, it would be -58% 

def get_nominal_drift(df_list, target_column):
    """This function calculates the nominal drift from the average of the aggregate population. It requires an input
    list of subset dataframes from the merged EPA+USDA+Redfin dataframe. It also requires the target column from 
    which the subset dataframes were created."""
    
    # Iterate through each subset dataframe to find the sectoral composition of emissions and add to nested dict  
    sectoral_composition_dict = {}
    for df in df_list: 
        composition = get_sectoral_baseline(df)
        key = df[target_column].iloc[0]
        sectoral_composition_dict[key] = composition

    # Given the composition for each subset dataframe, create a new nested dictionary, with the nominal drift 
    # from the baseline.
    nominal_drift_dict = {}
    
    #get the list of sectors 
    sector_set = list(sectoral_baseline.keys())
    
    # key = subset dataframe ID (ex. rural =1, 2...9), value = kv dict of sectoral CO ("mobile":12, "Fuel":13... ) 
    for k,v in sectoral_composition_dict.items():
        counter = 0 
        nominal_list = []
        
        # key = Sector, value = composition value 
        for key, value in v.items(): 
            
            # index is the sector: Ex "Mobile"
            index = sector_set[counter]
            
            # Find the drift from baseline (Baseline Composion - Current Subset's Composition)
            nominal_list.append({index: sectoral_baseline[index] - value})
            counter+=1
            
        # Add to sectoral drift of a given subset_df to the parent dictionary
        nominal_drift_dict[k] = nominal_list
    return(nominal_drift_dict)


def update_keys(input_dict, key_list):
    updated_dict = {}
    for i, key in enumerate(input_dict.keys()):
        updated_dict[key_list[i]] = input_dict[key]
    return(updated_dict)


# In[12]:


# Great, now we have the distribution of emissions for each urban/metropolitan designation. 
nominal_drift_dict = get_nominal_drift(dfs_sectoral,"Rural_Urban_Continuum_Code_2013") 

# At some point, we should map the Rural, urban codes to their implied geographical regions. We'll do that now since 
# it will make our plot easier to understand for the reader. While the USDA dataset did not provide an inherent 
# decoder, we accessed it at the link below, which corresponds with the codes 1-9 as ordered. 
# Link--> https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/
metro_decorder = ["Metro Pop. 1MM+", "Metro 1MM>Pop.>250k", "Metro 250k>Pop.>20k", 
            "Urban-Metro- 250k>Pop.>20k", "Urban not-Metro - 250k>Pop.>20k",
            "Urban-Metro- 20k>Pop.>5k", "Urban not-Metro - 20k>Pop.>5k",
            "Rural by Metro - Pop. <2.5k","Rural not by Metro - Pop. <2.5k"]


# In[13]:


#Finally let's built our sectoral chart
def sectoral_chart_build(nested_dict, subplot_x, subplot_y, title_list, axis_scale):
    """This function takes the nested dictionary of sectoral emissions for each subset dataframe. It requires the 
    x,y dimensions of each subplot, a list of titles that describe the unique population in each plot, and the 
    symmetric scaling value designed (+/-50 for example). The function is intentionally only suited for
    analysis with the merged EPA+USDA+Redfine dataframe."""
    
    # Create a list of charts in altair which will be returned for the user to select one or more of. 
    chart_list = []
    for i in range(len(nested_dict)):
        i += 1
        data_df = pd.DataFrame(nested_dict[i])
        
        #Re-shape the dataframe from wide-short to long-skinny and drop empty rows
        melted_df = data_df.melt(var_name='Sector', value_name='Value').dropna()

        # Define the order of sectors, to be consistent across plots
        sector_order = ['Commercial', 'Industrial Processes', 'Fuel Comb', 'Fires',
                        'Miscellaneous Non-Industrial NEC', 'Mobile']

        # Set y-axis title only for the leftmost plot
        y_title = 'Sectors' if (i == 1) else None

        # Create the chart, feeding the single subset dataframe (1/iteration), with bars on the x-axis separated 
        # nominally along the Y-axis, pull title from corresponding title, scale axis, and set y label
        chart = alt.Chart(melted_df).mark_bar().encode(
    x=alt.X('Value:Q', title=title_list[i-1], scale=alt.Scale(domain=[-axis_scale, axis_scale])),
            y=alt.Y('Sector:N', axis = alt.Axis(titleFontSize=20, title=y_title), 
                    sort=sector_order) if i == 1 else alt.Y('Sector:N', axis=None),
            color=alt.condition(
                alt.datum.Value > 0,
                alt.value('green'),  # Positive values in green
                alt.value('red')     # Negative values in red
            )
        ).properties(
            width=subplot_x,  # Set the width of each plot
            height=subplot_y  # Set the height of each plot
            
        )
        
        #add to aggregated chart list
        chart_list.append(chart)
    return(chart_list)
    

def display_chart(concat_chart,x_label, plot_title, title_shift):
    """This function provides customization for concatenated charts. By providing an input concatenated chart, 
    specifying the desired x label and title, as well as the shift to align the concatenated plot using 
    concatenation with an empty mark_text plot."""
    
    # Set empty footer chart with the xlabel title. This is done to enable a singular x-label across the 
    # whole plot set.
    footer_chart = alt.Chart().mark_text(
    ).encode(
        text=alt.value("")
    ).properties(
        title=alt.TitleParams(
            text=[x_label],
            baseline='bottom',
            orient='bottom',
            anchor='middle',
            fontWeight='bold',
            fontSize=20,
            dx=int(title_shift*.88) 
        )
    )

    
    concat_chart = concat_chart & footer_chart

    concat_chart = concat_chart.configure_axis(titleFontSize = 20,
        grid=False,).configure_view(strokeWidth=0).configure_title(
            fontSize=20  # Set the title font size
        ).configure_axis(
            labelFontSize=16  # Set the axis label font size
        ).properties(
        title=alt.TitleParams(plot_title,dx=title_shift))
    return(concat_chart)


# In[15]:


#Finally, let's go ahead and run this...
chart_list = sectoral_chart_build(nominal_drift_dict, 100, 200,metro_decorder, axis_scale=50 )    
concat_chart = (chart_list[0] | chart_list[1] | chart_list[2] | chart_list[3] | chart_list[-2])
display_chart(concat_chart,"Deviation", "Sectoral Emissions by Urbanization Level",350)


# In[ ]:


# Now for simplicity, we've only shown the major changes between population sizes from 1MM+ in the county to less than
# 2,500 people in the county. There's also opporunity to probe differences in emission between same population sizes 
# next to, or far away from metropolitan areas. 

# From the above graphic, we clearly see that emission contributions are drastically different based on the population 
# density of the area. For example, in high density areas, mobile emissions are significantly less than typical, but in 
# the most rural areas, mobile emissions are greatly in excess of the baseline composition.  

# Interestingly, commericial emissions from actions like gas stations, restaurant cooking, graphic design shops, etc, all 
# seem to be roughly the came contribution level for these 5 areas -- though they may be different for the 4 we have 
# not shown. 

# While it's outside the scope of this project to identify the cause of these emission differences, we could speculate 
# that the level of an area's requirement for personal transporation, in lieu of public transportation, is probably 
# driving this disparity (pun intended). 


# ### End of Notebook
# 
# That's the end of this notebook, in the next and final notebook, we'll tackle regional distributions of pollution across
# the United States. 
# 
# Next notebook: Multivariate Analysis: Geographic Choropleths
# 
# ---

# In[ ]:




