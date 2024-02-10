Purpose: 
This project uses USDA, EPA, and Redfin datasets to identify correlative trends between Carbon Monoxide pollution levels in the United States and regional wealth indicators (like income and property value). 

Agreement to Terms of Use and Copyright: 
By citing, using, distributing, or downloading this work (in part or in whole) you agree to adhere to the terms of use, defined in the "Terms of Use" file. You also agree
to honor copyright protections associated with this work. 

Installation Requirements: 
1. Programming Language & Version: Python 3.11.5
2. Python Packages:
   a) Pandas 2.0.3
   b) Numpy 1.24.3
   c) Matplotlib 3.7.2
   d) Seaborn 0.12.2 
   e) Altair 5.1.2 
   f) Plotly 5.9.0 
   g) Json 2.0.9
   h) Urllib 3.11 
   i) Sklearn 1.2.2

Disclaimers: 
1. Background: 
        Several key assumptions are made during this correlative analysis. The user should review and understand the key assumptions made throughout this project prior to infering any decisions based on 
        documented findings and results. The user must also refer to the Terms of Use to ensure their application of this study remains compliant with the stipulated requirements.

2. Core assumptions: 
   a) Data Quality: 
           This analysis assumes that the reported values from the data providers (USDA, EPA, and Redfin) are factually correct and free from error. However, the authors of this study make no gurantees regarding 
           the quality of the data supplied/used for this project. If data errors persist, the analysis presented in this study could mislead individuals by presential factually incorrect correlations. 
   b) Chronological Gaps: 
           Due to the nature of the datasets, chronological gaps exist. To correlate emission, income, labor force, and property value data, we must assume that U.S. Carbon Monoxide emissions are equivalent in year 2020 
           to those encountered in Year 2021. In addition we assume that rural-urban classifications for counties across the United States remain unchanged from their last assessment in the 2013 census (2023 census 
           rural-urban classification codes were not available at the start of this project). This is a significant because a global pandemic fundamentally shifted population behaviors in the beginng of 2020. For example,            at the onset of the pandemic in 2020, leisurely air travel significantly declined along with white-collar commuting activities. In addition, county-specific disasters could occur between 2020 and 2021 with                 profound impacts on both carbon monoxide emissions and local property values. Between 2013 and 2021, some regions may have enjoyed significant population growth promoting the location to a greater urbanization 
           level, which could render this variable inaccurate -- and counter-productive when conducting correlative emission analysis.    

   c) Generalizations: 
           To correlate pollution with wealth, we assume that carbon monoxide emission data alone are sufficient to generally characterize total air quality, whereas total air quality considers particulate 
           matter, sulfur dioxide, nitrous oxides, volatile organic compounds, greenhouse gas, and vaporized lead pollution levels. We also simplify consumer-level pollution to only consider residential and mobile 
           emission sources, when evaluating if wealthier households pollute more/less than their peers. Furthermore, we assume that median income and median home sale values are appropriate indiciators of regional 
           wealth, which doesn not consider affluent populations whose holdings generate limited or no taxable income. These broad generalizations could cause us to mischaracterize the true total pollution generation 
           rate of populations, or mischaracterize entire populations from assessment. For example, affluent retirement communities may exhibit low taxable income levels while actually maintaining a high level of net                 worth. Without additional information, the unique trends associated with this population are masked along with low-income populations depsite both groups enjoying very different qualities of life.    

3. Applicability concerns: 
        Data collected and analyzed during the period between 2020 and 2021 (such as pollution generation rates and income) may have been significantly impacted from the global COVID-19 pandemic, which temporarily shifted 
        many white collar industries to a hybrid or remote-based work system. The pandemic also forced the temporary closure of many businesses across the county, impacting income and unemployment rates. The user should 
        not extrapolate inferences during this time period to any other period. When updated pollution, income, and property value data are provided, a re-evaluation should be done to understand potential relations under 
        more standardized conditions. 
                    
