import pandas as pd
import numpy as np
from data_cleaning import clean_data
from summary_stats import summary_statistics
from summary_stats import churn_stats
from visuals import density_plot
from visuals import count_plot
from visuals import heat_map
from visuals import violin_plot
from visuals import check_multicollinearity
from logistical_reg import regression

# Read in the Data set and create a data frame - df || Set dependent variable
df = pd.read_csv('churn_clean.csv')
dependent_variable = 'Churn'
# Clean data
x_reference, x_analysis, y, one_hot_columns, binary_columns, categorical_columns, \
    continuous_columns, continuous_list = clean_data(df)
# continuous_list = continuous_columns.tolist()
# List non-numeric columns
non_numeric_columns = x_analysis.select_dtypes(exclude=[np.number]).columns

# Print non-numeric columns
if len(non_numeric_columns) > 0:
    print("Non-numeric columns:")
    print(non_numeric_columns)
else:
    print("All columns are numeric.")

# Get summary statistics
# summary_statistics(df, one_hot_columns, binary_columns, x_analysis)
# churn_stats(df, 'Churn')

# Plot uni-variate for categorical and continuous
density_plot(df, continuous_list)
count_plot(df, categorical_columns)

# Plot bi-variate for categorical and continuous against the dependant variable
heat_map(df, categorical_columns, dependent_variable)
violin_plot(df, continuous_list, dependent_variable)

check_multicollinearity(x_analysis)

regression(x_analysis, y)

# Retrieve 'dropped columns' and concatenate to df that was used for analysis and save to new CSV
result_df = pd.concat([x_reference, x_analysis], axis=1)
df.to_csv('churn_prepared.csv')
