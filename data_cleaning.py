import pandas as pd
import numpy as np
from visuals import plot_outliers


def clean_data(df):

    # Get a list of all the columns sorted by data type, for cleaning and analysis - Easier to look at the data this way
    # rather than in an .xlsx file and group columns by their data type
    dtype_groups = df.columns.groupby(df.dtypes)
    # Print out each data type and its columns of each data type
    for dtype, columns in dtype_groups.items():
        print(f"\nData Type: {dtype}")
        for column in columns:
            print(f"- {column}")

    # Print duplicated rows
    print("Duplicates:")
    print(df[df.duplicated()])

    # I ran a check for missing values and what columns had missing values, entered this line of code after running the
    # below code, but left the checking for missing values code there to verify that there are no missing values.
    df['InternetService'].fillna(df['InternetService'].mode()[0], inplace=True)
    # Get count of missing values and what column/s have missing values
    print("Missing Values")
    missing_values_count = df.isna().sum()
    missing_values_count = missing_values_count[missing_values_count > 0]
    print(missing_values_count)

    # Relabel the columns listed as item1...item8 with appropriate questions
    df.rename(columns={'Item1': 'Timely response',
                       'Item2': 'Timely fixes',
                       'Item3': 'Timely replacement',
                       'Item4': 'Reliability',
                       'Item5': 'Options',
                       'Item6': 'Respectful response',
                       'Item7': 'Courteous exchange',
                       'Item8': 'Evidence of active listening'},
              inplace=True)

    # Outliers
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    # Initializing a list to store outlier info
    outliers_info_list = []

    for col in numeric_cols.columns:
        data = numeric_cols[col]

        # Z-Score Method
        z_score = np.abs((data - data.mean()) / data.std())
        outliers_z_score = np.sum(z_score > 3)  # Typically, a Z-Score above 3 is considered as an outlier

        # IQR Method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers_iqr = np.sum((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))  # IQR Rule for Outliers

        # Storing info
        outliers_info_list.append({'Column': col,
                                   'Num_Outliers_ZScore': outliers_z_score,
                                   'Num_Outliers_IQR': outliers_iqr})

    # Creating DataFrame
    outliers_info = pd.DataFrame(outliers_info_list)

    # print(outliers_info)
    # Call the function
    plot_outliers(df)

    # Create two separate data frames, one with the dependant variable > y  and
    # one with all the independent variables > x - one that I will manipulate for analysis
    x = df.drop('Churn', axis=1)
    y = df['Churn']

    # Encoding
    # Create a group for columns that I want to keep around but do not want to use for analysis, then create
    columns_to_keep = ['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'Zip', 'Job', 'Population', 'Lat', 'Lng',
                       'City', 'State', 'County', 'Area', 'PaymentMethod', 'TimeZone', 'Tenure']
    # separate the data frames - columns to keep for later and columns for analysis
    x_reference = x[columns_to_keep]
    x_analysis = x.drop(columns=columns_to_keep)

    # create a binary mapping
    binary_mapping = {'Yes': 1, 'No': 0, 'DSL': 1, 'Fiber Optic': 0}

    # Select columns for binary mapping
    binary_columns = ['Techie', 'Tablet', 'Multiple', 'OnlineSecurity', 'OnlineBackup', 'Phone',
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                      'Port_modem', 'InternetService']

    # Apply binary mapping to selected columns
    for col in binary_columns:
        x_analysis[col] = x_analysis[col].map(binary_mapping)

    # Apply binary mapping to 'Churn', Dependent variable
    y = y.map(binary_mapping)

    # Select columns for one hot encoding
    one_hot_columns = ['Marital', 'Gender', 'Contract']

    # Create categories and groups of columns - Categorical and Continuous for ease of use in graphically viewing data
    categorical_columns = one_hot_columns + binary_columns

    continuous_list = [col for col in x_analysis.columns if col not in categorical_columns]
    print(continuous_list)

    # Check if the columns exist before encoding
    print("Columns before one-hot encoding:", df.columns)

    # Perform one-hot encoding
    x_analysis = pd.get_dummies(x_analysis, columns=one_hot_columns, drop_first=True)

    # Check if the encoding is done correctly
    print("Columns after one-hot encoding:", x_analysis.columns)

    # If you need to create a list of continuous columns after encoding
    continuous_columns = [col for col in x_analysis.columns if col not in categorical_columns]
    print()
    # Print DataFrame columns
    # print(df.columns)
    # print()
    # print(continuous_columns)

    return x_reference, x_analysis, y, one_hot_columns, binary_columns, categorical_columns, \
        continuous_columns, continuous_list
