# %% [markdown]
# ## Python Necessities 

# %% [markdown]
# ### Import Dependencies

# %%
##############################################################
# CLASS Project
# PYTHON IMPLEMENTATION: Introduction to Deep Learning and Neural Networks
# Course: CMPS3500
# Date: 12/06/24
# Student 1: Aldrin Amistoso
# Student 2: Jesse Garcia
# Student 3: Marc Angeles
# Student 4: Marvin Estrada
##############################################################

# General Packages
import math
import os
from pathlib import Path

# data handling libraries
import pandas as pd
import numpy as np
from tabulate import tabulate
import csv
from datetime import datetime
import time

# visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns

# extra libraries
import warnings
warnings.filterwarnings('ignore')

# Packages to support NN

# sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#tensorflow
import tensorflow as tf
from tensorflow import keras

# Keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

df = None

# %% [markdown]
# ### Load the data

# %%
# Load data from a folder called data within my project file
#  .. my_project
#     |
#     |___code
#     |   |
#     |   |__ CS3500_Starter_Notebook.ipynb
#     |
#     |___data
#         |
#         |__ credit_score.csv
#
#---------------------------------------------------------------

# # Get the current working directory
# current_dir = os.getcwd() 

# # Construct a path to the parent directory
# parent_dir = os.path.dirname(current_dir)

# # Access a file in the parent directory
# file_path = os.path.join(parent_dir, "data/credit_score_data.csv")

# # Load Credit Score data
# df = pd.read_csv(file_path) 

# global variable to track state
data_loaded = False
data_cleaned = False
model_trained = False

def loadData():
    global df, data_loaded
    """Load the credit score dataset and display summary statistics."""
    print("\nLoading and cleaning input data set:")
    print("************************************")
    start_time = time.time()
    
    # Get the current working directory
    current_dir = os.getcwd() 

    # Construct a path to the parent directory
    parent_dir = os.path.dirname(current_dir)

    # Access a file in the parent directory
    file_path = os.path.join(parent_dir, "data\credit_score_data.csv")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Script")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading training data set")
    
    try:
        # Load Credit Score data
        df = pd.read_csv(file_path) 
        total_columns = df.shape[1]
        total_rows = df.shape[0]
        end_time = time.time()
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total Columns Read: {total_columns}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total Rows Read: {total_rows}")
        data_loaded = True
        print(f"\nTime to load is: {round(end_time - start_time, 2)} seconds")
        
        return df
    except FileNotFoundError:
        print("Error: The file was not found. Please ensure the file path is correct and try again.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty. Please provide a valid dataset file and try again.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

# %% [markdown]
# ### Custom Functions For Data Cleaning

# %%
def describe_numerical_column(series, col_name):
    '''Describe a numerical column using describe function of series, report number of null values, display boxplots and histograms.
    Return min, max, IQR based outlier lower range and IQR based outlier upper range as a dictionary.'''
    # print(series.describe(), end = '\n\n')

    # print(f'Number of null values: {series.isnull().sum()}', '\n\n')
    
    # fig, ax = plt.subplots(2, 1, figsize = (8, 8), sharex = True)
    # sns.boxplot(series, orient = 'h', ax = ax[0])
    # ax[0].set_title(f'Distribution of {col_name}')
    # ax[0].tick_params(left = False, labelleft = False) 
    # sns.histplot(series, ax = ax[1])
    # ax[1].set_ylabel('Frequency')
    # ax[1].set_xlabel(col_name)
    # plt.show();

    q1, q3 = series.quantile([0.25, 0.75])
    IQR = q3 - q1
    
    return  {'Min. value': series.min(), 'Outlier lower range': q1 - 1.5 * IQR, 'Outlier upper range': q3 + 1.5 * IQR, 'Max. value': series.max()}

def summarize_numerical_column_with_deviation(data, num_col, group_col = 'Customer_ID', absolute_summary = True, median_standardization_summary = False):
    '''Summarize the numerical column and its median standardization based on customers using describe_numerical_column function.'''
    Summary_dict = {}
    
    if absolute_summary == True:
        # print(f'Column description for {num_col}:\n')
        Summary_dict[num_col] = describe_numerical_column(data[num_col], num_col)
        
    if median_standardization_summary == True:
        # if absolute_summary == True:
        #     print('\n')
        default_MAD = return_max_MAD(data, num_col, group_col)
        num_col_standardization = data.groupby(group_col)[num_col].apply(median_standardization, default_value = default_MAD)
        # print(f'Median standardization for {num_col}:\n')
        Summary_dict[f'Median standardization of {num_col}'] = describe_numerical_column(num_col_standardization, f'Median standardization of {num_col}')
        Summary_dict['Max. MAD'] = default_MAD
    return Summary_dict

def return_max_MAD(data, num_col, group_col = 'Customer_ID'):
    '''Return max value of median absolute devaition(MAD) from within the customers for num_col'''
    return (data.groupby(group_col)[num_col].agg(lambda x: (x - x.median()).abs().median())).max()
    
def validate_age(x):
    '''Check whether 8-months period age for a customer is logically valid or not'''
    diff = x.diff()
    if (diff == 0).sum() == 7:
        return True
    elif ((diff.isin([0, 1])).sum() == 7) and ((diff == 1).sum() == 1):
        return True
    else:
        return False
        
def median_standardization(x, default_value):
    '''Transform series or dataframe to its devaition from median with respect to Median absolute deviation(MAD) i.e. median standardization.'''
    med = x.median() 
    abs = (x - med).abs()
    MAD = abs.median()
    if MAD == 0:
        if ((abs == 0).sum() == abs.notnull().sum()): # When MAD is zero and all non-null values are constant in x
            return x * 0
        else:
            return (x - med)/default_value # When MAD is zero but all non-values are not same in x
    else:
        return (x - med)/MAD # When MAD is non-zero

def return_num_of_modes(x):
    '''Return number of modes in given series or dataframe'''
    return len(x.mode())

def return_mode(x):
    '''Return nan if no mode exists in given series or return minimum mode'''
    modes = x.mode()
    if len(modes) == 0:
        return np.nan
    return modes.min()

def forward_backward_fill(x):
    '''Perform forward fill then backward fill on given series or dataframe'''
    return x.fillna(method='ffill').fillna(method='bfill')

def return_mode_median_filled_int(x):
    '''Return back series by filling with mode(in case there is one mode) else fill with integer part of median'''
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(int(modes.median()))

def return_mode_average_filled(x):
    '''Return back series by filling with mode(in case there is one mode) else fill with average of modes'''
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(modes.mean())

def fill_month_history(x):
    '''Return months filled data for 8-months period'''
    first_non_null_idx = x.argmin()
    first_non_null_value = x.iloc[first_non_null_idx]
    return pd.Series(first_non_null_value + np.array(range(-first_non_null_idx, 8-first_non_null_idx)), index = x.index)

# %% [markdown]
# ### Custom Functions For Neural Networks Visuals

# %%
# Function to evaluate predicted vs test data categorical variables
def plot_prediction_vs_test_categorical(y_test, y_pred, class_labels):
    # Plots the prediction vs test data for categorical variables.

    # Args:
    #     y_test (array-like): True labels of the test data.
    #     y_pred (array-like): Predicted labels of the test data.
    #     class_labels (list): List of class labels.

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Calculates performance of multivariate classification model
def calculate_performance_multiclass(y_true, y_pred):
    # Calculates various performance metrics for multiclass classification.

    # Args:
    #     y_true: The true labels.
    #     y_pred: The predicted labels.

    # Returns:
    #     A dictionary containing the calculated metrics.

    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, and F1-score (macro-averaged)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics

# %% [markdown]
# ### Dataset pre-processing

# %% [markdown]
# ### Dropping unnecessary columns, data cleaning and correcting data types

# %% [markdown]
# **Note:**  
# Ideally data cleaning should be done in parallel to discussions with domain expert to understand what values are appropriate in the columns, can they be retrieved if missing and do the columns depend upon each other. Unfortunately, such kind of support is not available in this kaggle project and therefore, we will deal with the data as per our understanding approximately.

# %% [markdown]
# Looking at the dataset info, many of the columns in our dataset have null values within them, representing missing values. Also, some columns are not of the correct data type as per the data they hold, this means there might be some textual characters within the data indicating unclean data and maybe placeholders which describe non-existing data or missing data and therefore, that is not getting captured as null values but as strings. We need to identify these values and first change them to null values before we do any further pre-processing.

# %% [markdown]
# We will look at the columns one by one. Only columns which need some cleaning will be dealt with below.

# %% [markdown]
# ## Data Cleaning function

# %%
def dataCleaning():
    global df, data_loaded, data_cleaned
    try:
        if not data_loaded: # Check if data is loaded
            raise ValueError("Data has not been loaded. Please load the data first using option (1).")

        print("Process (Clean) data:")
        print("*********************")
        start_time = time.time()

        # Cleaning start
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Performing Data Clean Up")

        # #### 1. Customer ID

        # %%
        df['Customer_ID'].unique()

        # %%
        df['Customer_ID'].nunique()

        df['Customer_ID'].str.contains('CUS_0x').value_counts()

        # #### 2. Name
        df.drop(columns = ['Name'], inplace = True)

        # #### 3. Age
        df['Age'][~df['Age'].str.isnumeric()].unique() #extracting non-numeric textual data
        df['Age'] = df['Age'].str.replace('_', '')
        df['Age'][~df['Age'].str.isnumeric()].unique()
        df['Age'] = df['Age'].astype(int)

        # #### 4. SSN
        df.drop(columns = ['SSN'], inplace = True)

        # #### 5. Occupation
        df['Occupation'][df['Occupation'] == '_______'] = np.nan

        # #### 6. Annual Income
        df['Annual_Income'][~df['Annual_Income'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique() # using regex to find values which don't follow the patern of a float
        df['Annual_Income'] = df['Annual_Income'].str.replace('_', '')
        df['Annual_Income'][~df['Annual_Income'].str.fullmatch('([0-9]*[.])?[0-9]+')]
        df['Annual_Income'] = df['Annual_Income'].astype(float)

        # #### 7. Number of Loans

        df['Num_of_Loan'][~df['Num_of_Loan'].str.isnumeric()].unique()
        df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_', '').astype(int)

        # #### 8. Number of delayed payments
        temp_series = df['Num_of_Delayed_Payment'][df['Num_of_Delayed_Payment'].notnull()]

        temp_series[~temp_series.str.isnumeric()].unique()
        df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '').astype(float)

        # #### 9. Changed Credit Limit
        df['Changed_Credit_Limit'][~df['Changed_Credit_Limit'].str.fullmatch('[+-]?([0-9]*[.])?[0-9]+')].unique()
        df['Changed_Credit_Limit'][df['Changed_Credit_Limit'] == '_'] = np.nan 
        df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype(float)

        # #### 10. Credit Mix
        df['Credit_Mix'][df['Credit_Mix'] == '_'] = np.nan

        # #### 11. Outstanding debt
        df['Outstanding_Debt'][~df['Outstanding_Debt'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
        df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '')
        df['Outstanding_Debt'][~df['Outstanding_Debt'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
        df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(float)
        
        # #### 12. Amount Invested Monthly
        temp_series = df['Amount_invested_monthly'][df['Amount_invested_monthly'].notnull()]
        temp_series[~temp_series.str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
        df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace('_', '').astype(float)

        # #### 13. Payment Behaviour
        df['Payment_Behaviour'][df['Payment_Behaviour'] == '!@9#%8'] = np.nan

        # #### 14. Monthly Balance
        temp_series = df['Monthly_Balance'][df['Monthly_Balance'].notnull()]
        temp_series[temp_series.str.fullmatch('[+-]*([0-9]*[.])?[0-9]+') == False].unique()
        df['Monthly_Balance'][df['Monthly_Balance'] == '__-333333333333333333333333333__'] = np.nan
        df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
        df['Month'] = df['Month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8})
        df.sort_values(by = ['Customer_ID', 'Month'], ignore_index = True, inplace = True)
        df.drop(columns = 'ID', inplace = True)
        df.head(8)
        df_copy = df.copy()

        # #### 1. Age 
        df['Age'][(df['Age'] > 100) | (df['Age'] <= 0)] = np.nan 
        summary_age = summarize_numerical_column_with_deviation(df, 'Age', median_standardization_summary = True)
        df['Age'][df.groupby('Customer_ID')['Age'].transform(median_standardization, default_value = return_max_MAD(df, 'Age')) > 80] = np.nan
        df['Age'] =  df.groupby('Customer_ID')['Age'].transform(forward_backward_fill).astype(int)
        df.groupby('Customer_ID')['Age'].nunique().value_counts()

        df.groupby('Customer_ID')['Age'].agg(validate_age).value_counts()

        # #### 2. Occupation 
        df['Occupation'].isnull().sum()
        df.groupby('Customer_ID')['Occupation'].nunique().value_counts()
        df.groupby('Customer_ID')['Occupation'].count().value_counts()
        df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(forward_backward_fill)
        df['Occupation'].isnull().sum()

        # #### 3. Annual Income and monthly inhand salary
        summary_annual_income = summarize_numerical_column_with_deviation(df, 'Annual_Income', 'Customer_ID', True, False)

        summary_monthly_inhand_salary = summarize_numerical_column_with_deviation(df, 'Monthly_Inhand_Salary', 'Customer_ID', True, True)

        df.groupby(['Customer_ID', 'Monthly_Inhand_Salary'], group_keys = False)['Annual_Income'].transform(return_num_of_modes).value_counts()
        df[df.groupby(['Customer_ID', 'Monthly_Inhand_Salary'], group_keys = False)['Annual_Income'].transform(return_num_of_modes) == 2]
        df['Annual_Income'][df['Monthly_Inhand_Salary'].notnull()] = df[df['Monthly_Inhand_Salary'].notnull()].groupby(['Customer_ID', 'Monthly_Inhand_Salary'], group_keys = False)['Annual_Income'].transform(return_mode)

        summary_annual_income = summarize_numerical_column_with_deviation(df, 'Annual_Income', 'Customer_ID', True, False)

        df['Monthly_Inhand_Salary'] = df.groupby(['Customer_ID', 'Annual_Income'], group_keys = False)['Monthly_Inhand_Salary'].transform(forward_backward_fill)
        df['Monthly_Inhand_Salary'].isnull().sum()
        Annual_Income_deviation = df.groupby('Customer_ID', group_keys = False)['Annual_Income'].apply(lambda x: (x - x.median())/x.median())
        temp = Annual_Income_deviation[df['Monthly_Inhand_Salary'].isnull()]

        # print(temp.describe())
        df['Annual_Income'][df['Monthly_Inhand_Salary'].isnull()] = np.nan
        Annual_Income_deviation = df.groupby('Customer_ID', group_keys = False)['Annual_Income'].apply(lambda x: (x - x.median())/x.median())
        Annual_Income_deviation[Annual_Income_deviation > 500]
        df.iloc[[34042]]
        df[df['Customer_ID'].isin(['CUS_0x6079'])]

        df.loc[[34042], ['Annual_Income', 'Monthly_Inhand_Salary']] = np.nan

        df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].transform(forward_backward_fill)
        df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].transform(forward_backward_fill)
        summary_annual_income = summarize_numerical_column_with_deviation(df, 'Annual_Income', 'Customer_ID', True, False)
        summary_monthly_inhand_salary = summarize_numerical_column_with_deviation(df, 'Monthly_Inhand_Salary', 'Customer_ID', True, False)
        
        # #### 4. Number of Bank Accounts
        summary_num_bank_accounts = summarize_numerical_column_with_deviation(df, 'Num_Bank_Accounts', median_standardization_summary = True)
        summary_num_bank_accounts

        df['Num_Bank_Accounts'][df['Num_Bank_Accounts'] < 0] = np.nan
        df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Bank_Accounts')).value_counts()
        np.sort((df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Bank_Accounts'))).unique())[:10]
        df['Num_Bank_Accounts'][df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Bank_Accounts')).abs() > 2] = np.nan
        summary_num_bank_accounts = summarize_numerical_column_with_deviation(df, 'Num_Bank_Accounts', median_standardization_summary = True)
        df['Num_Bank_Accounts'] = df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(forward_backward_fill).astype(int)
        df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(pd.Series.diff).value_counts()

        df.groupby('Customer_ID')['Num_Bank_Accounts'].agg(lambda x: x.diff().sum()).value_counts()

        # #### 5. Number of credit cards
        summary_num_credit_cards = summarize_numerical_column_with_deviation(df, 'Num_Credit_Card', median_standardization_summary = True)
        summary_num_credit_cards
        df.groupby('Customer_ID')['Num_Credit_Card'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Credit_Card')).value_counts()
        np.sort((df.groupby('Customer_ID')['Num_Credit_Card'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Credit_Card'))).unique())[:10]

        df['Num_Credit_Card'][df.groupby('Customer_ID')['Num_Credit_Card'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Credit_Card')).abs() > 2] = np.nan

        summary_num_credit_cards = summarize_numerical_column_with_deviation(df, 'Num_Credit_Card', median_standardization_summary = True)
        df['Num_Credit_Card'] = df.groupby('Customer_ID')['Num_Credit_Card'].transform(forward_backward_fill).astype(int)
        df.groupby('Customer_ID')['Num_Credit_Card'].transform(pd.Series.diff).value_counts()

        df.groupby('Customer_ID')['Num_Credit_Card'].agg(lambda x: x.diff().sum()).value_counts()

        # #### 6. Interest Rate
        summary_interest_rate = summarize_numerical_column_with_deviation(df, 'Interest_Rate', median_standardization_summary = True)
        summary_interest_rate
        df.groupby('Customer_ID')['Interest_Rate'].nunique().value_counts()
        # What we observe is MAD is 0(since max. MAD is 0) for each customer. Thus, it is hard to look at median standardization and assess points using this. Lets try to look at deviation from median. Since interest rate is not a feature whose median should deviate too much in scale from customer to customer.
        deviation_from_median = df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: (x - x.median()))
        deviation_from_median.describe()
        deviation_from_median.value_counts()
        np.sort(deviation_from_median.unique())
        df['Interest_Rate'] = df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.median())
        df['Interest_Rate'].describe()

        # #### 7. Number of loans
        summary_num_of_loans = summarize_numerical_column_with_deviation(df, 'Num_of_Loan')
        df['Type_of_Loan'].isnull().sum()
        num_of_loans = df['Type_of_Loan'].str.split(', ').str.len()
        df['Num_of_Loan'][num_of_loans.notnull()] = num_of_loans[num_of_loans.notnull()]
        df['Num_of_Loan'][num_of_loans.isnull()].value_counts()

        # %%
        np.sort(df['Num_of_Loan'][num_of_loans.isnull()].value_counts().index)
        df['Num_of_Loan'][num_of_loans.isnull()] = 0

        # %%
        df['Num_of_Loan'] = df.groupby('Customer_ID')['Num_of_Loan'].transform(forward_backward_fill).astype(int)

        # %% [markdown]
        # What if we take one level difference at customer level? The difference can be 0, negative or positive but shouldn't be too high.

        # %%
        df.groupby('Customer_ID')['Num_of_Loan'].transform(pd.Series.diff).value_counts()

        # %% [markdown]
        # This means that number of loans remain same throughout the 8-months period for each customer.

        # %% [markdown]
        # #### 8. Type of loan

        # %%
        df['Type_of_Loan'].value_counts()

        # %% [markdown]
        # Here we see that the placeholder 'Not Specified' has been used as a way of indicating that the type of loan has not been specified by the customer. 

        # %%
        df['Type_of_Loan'].nunique()

        # %%
        df['Type_of_Loan'].isnull().sum()

        # %% [markdown]
        # Total 11408 null values. As noted earlier with number of loans these most probably represent no loans.

        # %% [markdown]
        # We can replace the same with our own placeholder for that - 'No Loan'.

        # %%
        df['Type_of_Loan'].fillna('No Loan', inplace = True)

        # %% [markdown]
        # Lets seen what and how many unique type of loans we have.

        # %%
        temp_series = df['Type_of_Loan']

        # %%
        temp_lengths = temp_series.str.split(', ').str.len().astype(int) # Number of loans

        # %%
        temp_lengths_max = temp_lengths.max()

        # %%
        for index, val in temp_lengths.items():
            temp_series[index] = (temp_lengths_max - val) * 'No Loan, ' + temp_series[index]

        # %%
        temp_series.head()

        # %%
        temp = temp_series.str.split(pat = ', ', expand = True)
        unique_loans = set()
        for col in temp.columns:
            temp[col] = temp[col].str.lstrip('and ')
            unique_loans.update(temp[col].unique())
        # print(unique_loans)

        # %%
        len(unique_loans)

        # %% [markdown]
        # There are total 8 unique type of loans, one placeholder for no specification of loan type and one placeholder added by us to specify there is no loan.

        # %% [markdown]
        # When we are working with tree based models usually they don't need the categorical columns to be encoded to numerical data type like we need for linear regression, logistic regression etc. as the model can handle these. But scikit learn uses CART algorithm and there is no functionality of using categorical variables directly and thus, for modelling with scikit-learn they need to be numerically encoded. Right now in this column we have 6260 unique categories which is too much to search for at one node. as at each node, a decision tree will look at all possible values of the categorical column to figure out what value produces the best split based on gini impurity or entropy decrease on splitting. We can do some pre-prcoessing on this column to split it into multiple columns. Intutively, it feels like the lastest loan should have high influence on your credit score because if the loan is heavy in nature then it might lead to poor credit score if unable to pay while if it is light then the credit score should remain almost same as before. We can split the column in following format: Latest loan1, latest loan2 etc. This way we will have 9 columns corresponding to maximum number of loans for any customer and each column can have maximum 10 categorical values corresponding to the unqiue loans calculated above. 

        # %% [markdown]
        # This approach will have following benefits:  
        # 1. Preserves the order of loans even after splitting.
        # 2. Easier to visualize and understand patterns since number of categories per column reduces.    
        # 3. Algorithm can focus more on information contained within the loan sequence. For example, if second last and third last loan contain critical information in classifying credit score than focussing on whole sequence of loans is not worthwhile and this inturn might lead to smaller decision trees and faster training compared to if we didn't split.  
        # 4. At each node for 9 of the splitted columns only total 90(9 columns * 10 categories) comparisons need to be made after one-hot encoding rather than 6260 comparisons for one non-splitted column.

        # %% [markdown]
        # Another possible approach could be to split columns as first loan, second loan etc. as a way of splitting but the pre-processing mentioned above feels more effective for now. We will maybe look at these two modeling startegies whn we do EDA and modeling.

        # %%
        temp.columns = [f'Last_Loan_{i}' for i in range(int(df['Num_of_Loan'].max()), 0, -1)]

        # %%
        temp.head()

        # %%
        df = pd.merge(df, temp, left_index = True, right_index = True)

        # %%
        df.head()
        df.drop(columns = 'Type_of_Loan', inplace = True)

        # %% [markdown]
        # #### 9. Delay from due date

        # %%
        summary_due_date = summarize_numerical_column_with_deviation(df, 'Delay_from_due_date', median_standardization_summary = True)

        # %%
        summary_due_date

        # %% [markdown]
        # The median standardization varies quite a bit going from -10 to 11. 

        # %%
        due_date_deviation = df.groupby('Customer_ID')['Delay_from_due_date'].transform(median_standardization, default_value = return_max_MAD(df, 'Delay_from_due_date'))

        # %% [markdown]
        # Looking at the fact that overall distribution of delay from due date is not too extreme and delay from due date can vary a lot as well unlike number of credit cards or number of bank accounts. We will move forward with the data as it is. Having a domain expert by your side would have helped make this more clearer.

        # %% [markdown]
        # #### 10. Number of delayed payments

        # %%
        summary_num_delayed_payments = summarize_numerical_column_with_deviation(df, 'Num_of_Delayed_Payment', median_standardization_summary = True)

        # %%
        summary_num_delayed_payments

        # %% [markdown]
        # Judging from median standardization, almost all of the values are same as median and this is leading to 0 median standardization. Median standardization should definitely should be like this and should be skewed in nature but its hard to assess what threshold to use without a domain expert. We will use the full column as a sample and judge based on that here.

        # %% [markdown]
        # The number of delayed payments can't be too much and can not be negative as well. We will set negative values and values greater than upper range of oultiers to null.

        # %%
        df['Num_of_Delayed_Payment'][(df['Num_of_Delayed_Payment'] > summary_num_delayed_payments['Num_of_Delayed_Payment']['Outlier upper range']) | (df['Num_of_Delayed_Payment'] < 0)] = np.nan

        # %%
        df['Num_of_Delayed_Payment'].isnull().sum()

        # %% [markdown]
        # There are 8382 null values. Lets observe the count of diff in between consecutive months and observe if we can identify some pattern exising there which can help us make some educated guess about the null values.

        # %%
        df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(pd.Series.diff).value_counts(normalize = True)

        # %% [markdown]
        # Around 45.6% of time it remains same across months but rest of the time it varies i.e more than 50% of the time it varies across months.

        # %%
        df[['Customer_ID', 'Num_of_Delayed_Payment']].head(40)

        # %% [markdown]
        # Looking at the data it looks like usually a single value repeats more often across months i.e. mode might be a suitable choice here. But first lets see that usually how many times the mode occurs for any customer.

        # %%
        # Ratio of frequency of mode and number of non-null data per customer
        temp = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(lambda x: (x == x.mode()[0]).sum()/x.notnull().sum()).value_counts(normalize = True)

        # %%
        temp[temp.index > 0.5].sum() # Idenitfying how many times the mode occurs in more than 50% of non-null data per customer

        # %% [markdown]
        # That is within given data for around 75.8% of the customers the mode occurs more than 50% of the time within 8-months period for whatever data we have available. This means the mode might be a suitable imputation here.

        # %% [markdown]
        # What if there are multiple modes per customer? Lets check the data if such thing exists.

        # %%
        df.groupby('Customer_ID')['Num_of_Delayed_Payment'].agg(lambda x: len(x.mode())).value_counts()

        # %% [markdown]
        # Mostly, we observe one mode but sometimes it can be more than one as well. What to do in multiple modes case? We can take some average or median of modes in that case, in case there is skewness within the modes, median would be a better guess and in case where medians come out to be floating point number we can just take the integer part as an approximation.

        # %%
        df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(return_mode_median_filled_int).astype(int)

        # %% [markdown]
        # Number of delayed payments is something that can vary from month to month its not like it has monotically increasing pattern or so which we can use as a sanity check. What we can try to observe is the relative devaition from median for this cleaned column.

        # %%
        summarize_numerical_column_with_deviation(df, 'Num_of_Delayed_Payment', median_standardization_summary = True)

        # %% [markdown]
        # #### 11. Changed credit limit

        # %%
        summary_changed_credit_limit = summarize_numerical_column_with_deviation(df, 'Changed_Credit_Limit', median_standardization_summary = True)

        # %% [markdown]
        # Credit card limit is dependent upon the users usage patterns. If the lender trusts the customer then it can increase also and if customer is late on payments, low activity etc. then the credit lmit can decrease as well. Thus, both negative and positive values are understandable.

        # %% [markdown]
        # The upper range for outliers for full column doesn't significantly deviate from the max value and its difficult to judge here what threshold should be placed on credit limit median standardization. Thus, we leave non-null values as it is for now.

        # %%
        df[['Customer_ID', 'Changed_Credit_Limit']].head(40)

        # %% [markdown]
        # Looking at the data usually the credit limit occurs with the same value across months i.e. the mode might be an appropriate value to imputate. Lets do some checks first though.

        # %%
        df.groupby('Customer_ID')['Changed_Credit_Limit'].agg(lambda x: len(x.mode())).value_counts()

        # %% [markdown]
        # Almost all the time only one mode appears. But sometimes two mode can occur as well, since this is a floating point type feature and there are only two mode values we will choose average of both which will be same as median in this case.

        # %%
        df['Changed_Credit_Limit'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(return_mode_average_filled)

        # %%
        df['Changed_Credit_Limit'].isnull().sum()

        # %% [markdown]
        # #### 12. Number of credit card inquiries

        # %%
        summary_num_credit_inquiries = summarize_numerical_column_with_deviation(df, 'Num_Credit_Inquiries', median_standardization_summary = True)
        df['Num_Credit_Inquiries'][(df['Num_Credit_Inquiries'] > summary_num_credit_inquiries['Num_Credit_Inquiries']['Outlier upper range']) | (df['Num_Credit_Inquiries'] < 0)] = np.nan

        # %%
        df['Num_Credit_Inquiries'].isnull().sum()

        # %% [markdown]
        # Lets look at some data.

        # %%
        df[['Customer_ID', 'Num_Credit_Inquiries']].head(40)

        # %% [markdown]
        # This a type of data which is monotically increasing in nature and thus should increase or remain same as months go on. Lets check that.

        # %%
        df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(pd.Series.diff).value_counts()

        # %% [markdown]
        # As expected it mostly either remains same or increases. In this case we can just use forward fill and backward fill to fill these nulls.

        # %%
        df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(forward_backward_fill).astype(int)

        # %% [markdown]
        # Lets do the check again.

        # %%
        df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(pd.Series.diff).value_counts()

        # %% [markdown]
        # #### 13. Credit Mix
        df['Credit_Mix'].isnull().sum()

        # %% [markdown]
        # A lot of null values present. We have already seen that during the 8-months period the type of loans and number of loans remain same so its fair to assume credit mix will also remain same. Lets check that with the given data.

        # %%
        df.groupby('Customer_ID')['Credit_Mix'].nunique().value_counts()

        # %% [markdown]
        # We can observe that one customer has only one type of credit mix only, throughout the 8-months period apart from null values. We can just use forward fill and bacward fill to achieve the desired goal.

        # %%
        df['Credit_Mix'] = df.groupby('Customer_ID')['Credit_Mix'].transform(forward_backward_fill)

        # %%
        df['Credit_Mix'].isnull().sum()

        # %% [markdown]
        # #### 14. Outstanding debt

        # %%
        summary_outstanding_debt = summarize_numerical_column_with_deviation(df, 'Outstanding_Debt', median_standardization_summary = True)

        # %% [markdown]
        # All the values in median standardization are coming out to be zero. Is the outstanding debt constant for each customer across months after ignoring nulls?

        # %%
        df.groupby('Customer_ID')['Outstanding_Debt'].nunique().value_counts()

        # %% [markdown]
        # The column looks ok from the distribution perspective and there are no nulls present.

        # %% [markdown]
        # #### 15. Credit Utilization ratio

        # %%
        summary_credit_utilization_ratio = summarize_numerical_column_with_deviation(df, 'Credit_Utilization_Ratio', median_standardization_summary = True)

        # %% [markdown]
        # Judging from both the graphs its hard to put a threshold on median standardization of credit utilization ratio without a domin expert and also, the distribution of the column as a whole looks decent enough to not touch it further.

        # %% [markdown]
        # #### 16. Credit History Age

        # %%
        df[['Customer_ID', 'Credit_History_Age']].head(40)

        # %%
        df['Credit_History_Age'].isnull().sum()

        # %% [markdown]
        # There are 9030 null values. Looking at the data it is of the format - '{Year} Years and {Months} Months'. Using str functions of pandas series, we can extract these two data values i.e. year and months. We will then combine them in a single column as total months because both the year data and month data can be easily extracted from total months so there will be no loss of information and we will be able to reduce one feature from our dataset.

        # %%
        df[['Years', 'Months']] = df['Credit_History_Age'].str.extract('(?P<Years>\d+) Years and (?P<Months>\d+) Months').astype(float)
        df[['Years', 'Months']].describe()
        df['Credit_History_Age'] = df['Years'] * 12 + df['Months']

        # %%
        df.drop(columns = ['Years', 'Months'], inplace = True)

        df['Credit_History_Age'].isnull().sum()
        df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(fill_month_history).astype(int)

        # #### 17. Payment of minimum amount
        df['Payment_of_Min_Amount'].value_counts()

        df.groupby(['Customer_ID'])['Payment_of_Min_Amount'].nunique().value_counts()

        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0, 'NM': np.nan})

        df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(pd.Series.diff).value_counts()

        df['Payment_of_Min_Amount'] = df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(lambda x: x.fillna(x.mode()[0]))
        df['Payment_of_Min_Amount'].isnull().sum()
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({1: 'Yes', 0: 'No'})

        # #### 18. Total EMI per month
        summary_total_emi_per_month = summarize_numerical_column_with_deviation(df, 'Total_EMI_per_month', median_standardization_summary = True)
        summary_total_emi_per_month

        deviation_total_emi = df.groupby('Customer_ID')['Total_EMI_per_month'].transform(median_standardization, default_value = return_max_MAD(df, 'Total_EMI_per_month'))

        df['Total_EMI_per_month'][deviation_total_emi > 10000] = np.nan
        summary_total_emi_per_month = summarize_numerical_column_with_deviation(df, 'Total_EMI_per_month', median_standardization_summary = True)
        df['Total_EMI_per_month'][(df['Total_EMI_per_month'] > summary_total_emi_per_month['Total_EMI_per_month']['Outlier upper range'])] = np.nan
        df['Total_EMI_per_month'].isnull().sum()
        df.groupby('Customer_ID')['Total_EMI_per_month'].nunique().value_counts()
        deviation_total_emi = df_copy.groupby('Customer_ID', group_keys = False)['Total_EMI_per_month'].transform(median_standardization, default_value = return_max_MAD(df_copy, 'Total_EMI_per_month'))

        # %%
        temp = (deviation_total_emi[df.groupby('Customer_ID')['Total_EMI_per_month'].transform(pd.Series.nunique) == 0])

        # %% [markdown]
        # Looking at this, only one value looks absurdly big. The rest of the median standardization's could even be considered ok for now.

        # %%
        temp[temp > 80]

        # %% [markdown]
        # Lets convert this value to null and feed it back to the dataset.

        # %%
        temp[79370] = np.nan
        df['Total_EMI_per_month'][temp.index] = temp

        # %%
        summarize_numerical_column_with_deviation(df, 'Total_EMI_per_month', median_standardization_summary = True)

        # %% [markdown]
        # Now the data looks more appropriate compared to before. There are still 4420 null values which need to be handled here.

        # %% [markdown]
        # The rest of the null values can be filled using forward and backward fill as the EMI's should be highly dependent upon previous month.

        # %%
        df['Total_EMI_per_month'] = df.groupby('Customer_ID')['Total_EMI_per_month'].transform(forward_backward_fill)

        # %%
        df['Total_EMI_per_month'].isnull().sum()

        # %% [markdown]
        # #### 19. Amount Invested Monthly

        # %%
        summary_amount_invested_monthly = summarize_numerical_column_with_deviation(df, 'Amount_invested_monthly', median_standardization_summary = True)

        # %% [markdown]
        # Some values in amount invested monthly are too extreme compared to the rest of the data and thus, can be removed considering them to be erroneous before we do further processing.

        # %%
        df['Amount_invested_monthly'][df['Amount_invested_monthly'] > 8000] = np.nan

        # %% [markdown]
        # Lets check the distribution again.

        # %%
        summary_amount_invested_monthly = summarize_numerical_column_with_deviation(df, 'Amount_invested_monthly', median_standardization_summary = True)

        # %% [markdown]
        # Looks like power law distribution, hopefully these are not erroneous values. Lets leave these non-null values as it is for now. Null values still need to be handled.

        # %%
        df.groupby('Customer_ID')['Amount_invested_monthly'].transform(return_num_of_modes).value_counts()

        # %% [markdown]
        # Lets choose the median of values as a decent approximation for null values.

        # %%
        df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.fillna(x.median()))

        # %% [markdown]
        # #### 20. Payment Behaviour
        df['Payment_Behaviour'].isnull().sum()

        # %% [markdown]
        # 7600 of null values present.

        # %%
        df.groupby('Customer_ID')['Payment_Behaviour'].nunique().value_counts()

        # %%
        df.groupby('Customer_ID')['Payment_Behaviour'].agg(return_num_of_modes).value_counts()

        # %% [markdown]
        # The number of modes vary, if the number of mode is 1 then we can use that for imputation else forward fill and backward fill can be used.

        # %%
        df['Payment_Behaviour'] = df.groupby('Customer_ID')['Payment_Behaviour'].transform(lambda x: return_mode(x) if len(x.mode()) == 1 else forward_backward_fill(x))

        # %%
        df['Payment_Behaviour'].isnull().sum()

        # %% [markdown]
        # #### 21. Monthly Balance

        # %%
        summary_monthly_balance = summarize_numerical_column_with_deviation(df, 'Monthly_Balance', median_standardization_summary = True)

        # %% [markdown]
        # There are 1209 null values. Looking at the column as a whole the distribution looks ok and considering the fact that we can't decide exactly on a threshold on median standardization without domain expertise. We will leave the non-null values as it for now.

        # %%
        df.groupby('Customer_ID')['Monthly_Balance'].nunique().value_counts()

        # %% [markdown]
        # Since there might be skewness within the data we can use median to fill the null values.

        # %%
        df['Monthly_Balance'] = df.groupby('Customer_ID')['Monthly_Balance'].transform(lambda x: x.fillna(x.median()))

        # %% [markdown]
        # ### Deleting unnecessary columns

        # %%
        df.columns

        # %% [markdown]
        # Month column is not needed anymore and can be dropped. We will keep customer id as it is for now so that it can be used later on when doing train-test splits.

        # %%
        df.drop(columns = ['Month'], inplace = True)

        # %%
        df = df.sample(frac = 1) #shuffle data

        # %% [markdown]
        # ### Rearranging the columns

        # %%
        df.columns

        # %%
        df = df.loc[:, ['Customer_ID', 'Age', 'Occupation', 'Annual_Income',
            'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
            'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
            'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
            'Credit_Utilization_Ratio', 'Credit_History_Age',
            'Payment_of_Min_Amount', 'Total_EMI_per_month',
            'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance', 'Last_Loan_9', 'Last_Loan_8', 'Last_Loan_7',
            'Last_Loan_6', 'Last_Loan_5', 'Last_Loan_4', 'Last_Loan_3',
            'Last_Loan_2', 'Last_Loan_1',
            'Credit_Score']]

        # Total rows after cleaning
        total_rows_after_cleaning = len(df)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Total Rows after cleaning is: {total_rows_after_cleaning}")
        # Export the cleaned data
        parent_dir = os.path.dirname(os.getcwd())
        file__out_path = os.path.join(parent_dir, "data\Credit_score_cleaned_data.csv")
        df.to_csv(file__out_path, index = False)
        data_cleaned = True
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nTime to process is: {processing_time:.2f} seconds")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Error during data cleaning: {e}")

    return df


# %% [markdown]
# ## Neural Network Code

# %% [markdown]
# ### Selecting Input For Neural Networks
# Drop or add columns that for the model.
# This is where you control your input for the model.

# %%
def trainNN():
    global df, X_train, X_test, y_train, y_test, model, encoder, data_loaded, data_cleaned, model_trained
    try:
        if not data_loaded:  # Check if data is loaded
            print("Error: Data has not been loaded. Please load the data first using option (1).")
            return
        if not data_cleaned:
            print("Error: Data has not been cleaned. Please clean the data first using option (2).")
            return

        print("Train NN:")
        print("********")

        start_time = time.time()

        # target and features
        target = ['Credit_Score']
        continuous_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card', 'Interest_Rate', 'Credit_Utilization_Ratio', 'Credit_History_Age','Monthly_Balance', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Num_of_Delayed_Payment', 'Delay_from_due_date', 'Changed_Credit_Limit', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Num_Bank_Accounts', 'Num_of_Loan'] 
        categorical_features = ['Credit_Mix', 'Payment_Behaviour','Last_Loan_5', 'Last_Loan_4', 'Last_Loan_3','Last_Loan_2', 'Last_Loan_1', 'Payment_of_Min_Amount', 'Payment_of_Min_Amount', 'Last_Loan_9','Last_Loan_8', 'Last_Loan_7', 'Last_Loan_6', 'Occupation']

        # Encoding features and target
        encoder = OneHotEncoder(handle_unknown='ignore')
        le = LabelEncoder()

        # Encoding categorical features
        encoded_features = encoder.fit_transform(df[categorical_features])

        # Convert the encoded data back to a DataFrame:
        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(categorical_features))

        # joining dataframes 
        df = pd.concat([df, encoded_df], axis=1)
        # Encoding categorical features
        encoded_target = encoder.fit_transform(df[target])

        # Convert the encoded data back to a DataFrame:
        encoded_target_df = pd.DataFrame(encoded_target.toarray(), columns=encoder.get_feature_names_out(target))

        # joining dataframes 
        df = pd.concat([df, encoded_target_df], axis=1)
        # Constructing dataframe for modeling
        features_for_model = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Card', 'Interest_Rate'
                            , 'Credit_Utilization_Ratio', 'Credit_Mix_Bad', 'Credit_Mix_Good', 'Credit_Mix_Standard'
                            , 'Last_Loan_3','Last_Loan_2', 'Last_Loan_1', 'Credit_History_Age','Monthly_Balance'
                            , 'Payment_Behaviour', 'Num_Credit_Inquiries','Last_Loan_5', 'Last_Loan_4', 'Outstanding_Debt'
                            , 'Num_of_Delayed_Payment', 'Delay_from_due_date', 'Payment_of_Min_Amount', 'Changed_Credit_Limit'
                            , 'Total_EMI_per_month', 'Amount_invested_monthly', 'Num_Bank_Accounts', 'Num_of_Loan', 'Last_Loan_9'
                            ,'Last_Loan_8', 'Last_Loan_7', 'Last_Loan_6', 'Occupation_Accountant', 'Occupation_Architect'
                            , 'Occupation_Developer', 'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Entrepreneur'
                            , 'Occupation_Journalist', 'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Mechanic'
                            ,'Occupation_Media_Manager' , 'Occupation_Musician', 'Occupation_Scientist', 'Occupation_Teacher'
                            , 'Occupation_Writer'
                            ] 

        target_features = ['Credit_Score_Good', 'Credit_Score_Poor', 'Credit_Score_Standard']

        # Defining data sets
        X = encoded_features.toarray()
        y = encoded_target.toarray()

        # Basic train-test split
        # 80% training and 20% test 
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.20, random_state=42)

        # Create network topology
        model = keras.Sequential()

        # Adding input model --> 24 input layers
        model.add(Dense(46, input_dim = X_train.shape[1], activation = 'relu'))

        # Adding hidden layer 
        model.add(keras.layers.Dense(1000, activation="relu"))
        model.add(keras.layers.Dense(512, activation="relu"))
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(64, activation="relu"))

        # # output layer
        # # For classification tasks, we generally tend to add an activation function in the output ("sigmoid" for binary, and "softmax" for multi-class, etc.).
        model.add(keras.layers.Dense(3, activation="softmax"))

        # print(model.summary())

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        model.fit(X_train, y_train, epochs = 30, batch_size = 120)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred = y_pred.argmax(axis=1)
        y_test_decoded = y_test.argmax(axis=1)

        # Calculate metrics
        accuracy = test_acc
        precision = precision_score(y_test_decoded, y_pred, average='weighted')
        recall = recall_score(y_test_decoded, y_pred, average='weighted')
        f1 = f1_score(y_test_decoded, y_pred, average='weighted')
        confusion_matrix_result = confusion_matrix(y_test_decoded, y_pred)

        # Logging metrics
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Model Accuracy: {accuracy:.4f}")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Model Precision: {precision:.4f}")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Model Recall: {recall:.4f}")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Model f1_score: {f1:.4f}")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Model Confusion Matrix: \n{confusion_matrix_result}")
        model_trained = True
        end_time = time.time()
        print(f"\nTime to process: {end_time - start_time:.2f} seconds")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Error during model training: {e}")
    



# %% [markdown]
# # Predictions

# %%
def predictions():
    global X_test, y_test, model, data_loaded, data_cleaned, model_trained
    try:
        # Check if the data was loaded
        if not data_loaded:
            raise ValueError("Data has not been loaded. Please load the data first using option (1).")

        # Check if the data was cleaned
        if not data_cleaned:
            raise ValueError("Data has not been cleaned. Please clean the data first using option (2).")

        # Check if the model was trained
        if not model_trained:
            raise ValueError("The model has not been trained. Please train the model first using option (3).")

        print("Generate Predictions:")
        print("********************")

        start_time = time.time()

        # Generate predictions
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Generating predictions using the trained Neural Network")
        predictions = model.predict(X_test)

        # Decode true labels and predictions
        y_predicted = predictions.argmax(axis=1)  # Predicted class indices

        # Map numeric predictions to human-readable labels
        label_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        y_predicted_labels = [label_mapping[label] for label in y_predicted]

        # Retrieve corresponding Customer IDs
        customer_ids = df.loc[df.index[X_test.shape[0] * -1:], 'Customer_ID'].values  # Select IDs from test set

        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            "ID": customer_ids,
            "Credit_Score": y_predicted_labels
        })
        parent_dir = os.path.dirname(os.getcwd())
        file_out_path = os.path.join(parent_dir, "data\predictionClassProject3.csv")
        predictions_df.to_csv(file_out_path, index=False)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Predictions saved to 'predictionClassProject1.csv'")

        end_time = time.time()
        print(f"\nTime to process: {end_time - start_time:.2f} seconds")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Error during predictions: {e}")


# %% [markdown]
# # Main Menu

# %%
def mainMenu():
    global df, X_test, y_test, model, encoder
    while True:
        print("\n===============================================")
        print("Menu:")
        print("(1) Load data")
        print("(2) Process (Clean) data")
        print("(3) Train NN")
        print("(4) Generate Predictions")
        print("(5) Quit")
        print("===============================================")
        
        choice = input("Select Option: ")
        
        if choice == '1':
            loadData()
        
        elif choice == '2':
            dataCleaning()
                
        elif choice == '3':
            trainNN()
        
        elif choice == '4':
            predictions()
        
        elif choice == '5':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid option. Please select a valid option.")

if __name__ == "__main__":
    mainMenu()




