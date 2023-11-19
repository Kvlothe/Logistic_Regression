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
from logistical_reg import feature_selection_rfe

# Read in the Data set and create a data frame - df || Set dependent variable
df = pd.read_csv('churn_clean.csv')
dependent_variable = 'Churn'
# Clean data
x_reference, x_analysis, y, one_hot_columns, binary_columns, categorical_columns, \
    continuous_columns, continuous_list, df_analysis = clean_data(df)
# continuous_list = continuous_columns.tolist()
# List non-numeric columns
non_numeric_columns = x_analysis.select_dtypes(exclude=[np.number]).columns

# Print non-numeric columns
# if len(non_numeric_columns) > 0:
#     print("Non-numeric columns:")
#     print(non_numeric_columns)
# else:
#     print("All columns are numeric.")

# Get summary statistics
# summary_statistics(df, one_hot_columns, binary_columns, df_analysis)
# churn_stats(df, 'Churn')

# Plot uni-variate for categorical and continuous
# density_plot(df, continuous_list)
# count_plot(df, categorical_columns)

# Plot bi-variate for categorical and continuous against the dependant variable
# heat_map(df, categorical_columns, dependent_variable)
# violin_plot(df, continuous_list, dependent_variable)

# Check for multicollinearity
check_multicollinearity(x_analysis)
# Run initial regression model with all independent variables
regression(x_analysis, y)
# Feature selection using RFE
selected_features = feature_selection_rfe(x_analysis, y, n_features_to_select=5)
# create a data frame with the selected features from my analysis data frame -
# to ensure one-hot encoded columns are present and pass selected features from RFE to the regression model
x_selected = x_analysis[selected_features]
regression(x_selected, y)

# Save the prepared data
x_analysis.to_csv('churn_prepared.csv')
