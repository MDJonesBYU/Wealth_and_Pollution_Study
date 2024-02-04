#importing required packages: 
import matplotlib.pyplot as plt 

# Importing py files
from Basic_data_structure import *
from Data_manipulation import *


def hist_plot_base (bin_list, level="STATE"):
    """This function creates 4 histograms to evaluate the distribution of income, property values, unemployment, and 
    aggregate residential emissions at either the state level or the county county."""
    
    if level == "STATE": 
        df1, df2, df3, df4 = df_USDA_State, df_Redfin_State, df_USDA_State, df_emissions_State
    elif level == "COUNTY":
        df1, df2, df3, df4 = df_USDA_County, df_Redfin_County, df_USDA_County, df_emissions_County        
          
    dfs = [df1, df2, df3, df4]
    
    #Set the figure object, while providing a numpy array of axis (our 2x2 matrix)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["Income", "Property Value", "Unemployment Rates", "Emissions"]
    xlabels = ["USDx1k", "USDx100k", "Percent", "TPYx100k"]
    colors = ['skyblue', 'salmon', 'lightgreen', 'gold']
    columns = ["Median_Household_Income_2021", "median_sale_price", "Unemployment_rate_2021", "EMISSIONS"]
    divisor = [1e3, 1e5, 1, 1e5]
    
    # We're going to iterate through each of the 4 dataframes, and assign the proper title, label, coloring, and 
    # scaling factor using the zip method on our flattened list of subplots. 
    for ax, df, title, xlabel, bins, color, column, div in zip(axs.flat, dfs, titles, xlabels,
                                                               bin_list, colors, columns, divisor):
        ax.hist(df[column] / div , bins=bins, color=color, edgecolor='black')
        ax.set_title(f'Distribution of {title}')
        ax.set_ylabel('Count')
        ax.set_xlabel(xlabel)
        if level=="COUNTY" and title == "Emissions": 
            axs[1, 1].set_xlim(0, 0.6)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Show plot
    plt.show()


def get_extremes(df, column): 
    """filter dataframe to the top 5 and lowest five entries. We'll do this by creating a copy of the original"""
    top_vals = df.nlargest(5, column)
    bot_vals = df.nsmallest(5, column)

    # bring the two back together for our top and bottom 5
    output_df = pd.concat([top_vals, bot_vals])
    return(output_df)


def bar_plotter(dfs):
    
    # Set the figure object, while providing a numpy array of axis (our 2x2 matrix)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["Income", "Property Value", "Unemployment Rate", "Emission"]
    ylabels = ["USDx1k", "USDx100k", "Percent", "TPYx100k"]
    colors = ['skyblue', 'salmon', 'lightgreen', 'gold']
    columns = ["Median_Household_Income_2021", "median_sale_price", "Unemployment_rate_2021", "EMISSIONS"]
    divisor = [1e3, 1e5, 1, 1e5]
    states = ["State", "state_code", "State", "STATE"]
    # We're going to iterate through each of the 4 dataframes, and assign the proper title, label, coloring, and 
    # scaling factor using the zip method on our flattened list of subplots. 
    for ax, df, title, ylabel, color, column, div, state in zip(axs.flat, dfs, titles, ylabels,
                                                         colors, columns, divisor, states):
        #set indices for the states (our x range)
        indices = np.arange(len(df)) 
        ax.bar(indices, df[column] / div, color=color, edgecolor='black') 
        # Set x-ticks at bar positions
        ax.set_xticks(indices)  
        # Set x-tick labels to the state
        ax.set_xticklabels(df[state])  
        ax.set_title(f'Comparison of Top and Bottom 5: {title}s')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("State")
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    




