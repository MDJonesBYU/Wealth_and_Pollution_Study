# Multivariate Exploratory Data Analysis: Combo Charts and Heat Maps

#importing required packages: 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Importing py files
from Basic_data_structure import *
from Data_manipulation import *
from Univariate_EDA import get_extremes


def extremes_bar_plotter(dfs):
    
    # Set the figure object, while providing a numpy array of axis (our 2x2 matrix)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["Pollution to Income Ratio -- County Level", "Pollution to Income Ratio -- State Level", 
              "Pollution to Property Value Ratio -- County Level", "Pollution to Property Value Ratio -- State Level"]
    ylabels = ["P2I Ratio", "P2I Ratio", "P2PV Ratio", "P2PV Ratio"]
    colors = ['skyblue', 'salmon', 'lightgreen', 'gold']
    columns = ["PI_ratio", "PI_ratio", "PPV_ratio", "PPV_ratio"]
    divisor = [1, 1, 1, 1]
    locations = ["COUNTY", "STATE", "COUNTY", "STATE"]
    
    # We're going to iterate through each of the 4 dataframes, and assign the proper title, label, coloring, and 
    # scaling factor using the zip method on our flattened list of subplots. 
    for ax, df, title, ylabel, color, column, div, location in zip(axs.flat, dfs, titles, ylabels,
                                                         colors, columns, divisor, locations):
        #set indices for the states (our x range)
        indices = np.arange(len(df)) 
        bars = ax.bar(indices, df[column] / div, color=color, edgecolor='black',width=.8) 
        # Set x-ticks at bar positions
        ax.set_xticks(indices)  
        # Set x-tick labels to the state
        ax.set_xticklabels(df[location], rotation=45)  
        ax.set_title(f'{title}')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("State")
        
        ax2 = ax.twinx()
        ax2.plot(indices, df["Civilian_labor_force_2021"]/100000, 'ko--')
        ax2.set_ylabel('Population x 100k', color='k')        
        
        for bar, value in zip(bars, df[column] / div):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 1, height, f'{value:.1e}', ha='center', va='bottom',fontsize=10, 
                    rotation=45)        

    max_height = ax.get_ylim()[1]
    height_adjustment = [40,5,30, 1.3]
    max_len = ax.get_xlim()[1]
    len_adjustment = [1.05 for i in range(4)]
    counter = 0
    for ax in axs.flat:
        ax.set_ylim(top=max_height * height_adjustment[counter])
        ax.set_xlim(right=max_len * len_adjustment[counter])
        counter +=1
    plt.subplots_adjust(hspace=0.7, wspace=0.4)


def plot_correlations(df, include_continuum=True):
    """This function creates a correlation plot of the merged dataframe between per person CO emissions and variables
    like income, unemployment, property value, level of urbanization (Rural urban contiuum code). The user inputs
    whether or not the entire continuum is included to adjust the number of sqaures in the heatmap."""
    label_encoder_y = LabelEncoder()
    df['Encoded Emissions per Person'] = label_encoder_y.fit_transform(df['Emissions per Person'])

    correlation_list = ["Emissions per Person", "Civilian_labor_force_2021", 
                               "Median_Household_Income_2021", "Unemployment_rate_2021", "median_sale_price",
                               "Rural_Urban_Continuum_Code_2013"]
    
    if include_continuum == False:
        correlation_list = correlation_list[0:-1]
        print(correlation_list)
    elif include_continuum != True:
        raise ValueError("include_continuum must be boolean True or False") 
    
    # Make a list of columns to correlate to the Emissions per person value
    corr_cols = df[correlation_list]
    
    # Run correlation calculations on the dataframe
    corr = corr_cols.corr()

    plt.figure(figsize=(18, 10))

    # Plot the correlation heatmap using seaborn, cividis for colorblind, and setting to 2 decimal places.
    sns.heatmap(corr, annot=True, cmap='cividis', fmt='.2f')

    #Make x labels reader friendly
    custom_ticklabels = ["Emissions \nper Person", "Civilian \nLabor Force", 
                          "Median \nHousehold Income", "Unemployment \nRate", "Median \nHome Price",
                          "Rural vs. Urban \nContinuum"]

    if include_continuum == False:
        custom_ticklabels = custom_ticklabels[0:-1]
    
    plt.xticks(ticks=range(len(custom_ticklabels)), labels=custom_ticklabels, fontsize=14, ha='left', rotation=0)
    plt.yticks(ticks=range(len(custom_ticklabels)), labels=custom_ticklabels, fontsize=14, va='center')

    # Set x-label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if include_continuum == False: 
        _string = "for RUC Code {}".format(int(df["Rural_Urban_Continuum_Code_2013"].iloc[0]))
    else: 
        _string = ""
    # Add a title
    plt.title("Correlation of CO Emissions (per person) {}".format(
        _string), fontsize=22)

    plt.show()
    return() 


