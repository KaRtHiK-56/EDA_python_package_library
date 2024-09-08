import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,RobustScaler
from scipy import stats 
from scipy.stats import skew,kurtosis
import warnings
warnings.filterwarnings('ignore')

path = input("Please enter the path of your csv file: ")

df = pd.read_csv(path)

class Inspection():

    def __init__(df):
        df = df
    
    def inspect():
        
        size = df.size
        shape = df.shape
        dimentions = df.ndim
        info = df.info()
        summary = df.describe()
        columns = df.columns
        n_col = len(columns)
        types = df.dtypes
        duplicated = df.duplicated().sum()
        p_dup = duplicated / (len(df)*100)
        missing = df.isnull().sum()
        p_miss = missing / (len(df)*100)
        numeric = df.select_dtypes(include=['int64','float64'])
        n_num = len(numeric)
        categoical = df.select_dtypes(exclude=['int64','float64'])
        n_cat = len(categoical)
        datetime = df.select_dtypes(include=['datetime64'])
        corr = df.corr(numeric_only=True)

        for col in numeric.columns:
            print(f"\nProcessing column for calculating skewness: {col}")

            skewness_value = skew(numeric[col].dropna())  # Dropping NaN to avoid skew calculation issues
            print(f"Skewness of '{col}': {skewness_value}")

            if skewness_value > 1:
                print(f"'{col}' is Highly Positively Skewed(Right Skewed).")
            elif 0.5 < skewness_value <= 1:
                print(f"'{col}' is Moderately Positively Skewed.")
            elif -0.5 <= skewness_value <= 0.5:
                print(f"'{col}' is Approximately Symmetric.")
            elif -1 <= skewness_value < -0.5:
                print(f"'{col}' is Moderately Negatively Skewed(Left Skewed).")
            else:
                print(f"'{col}' is Highly Negatively Skewed.")
        
        for col in numeric.columns:
            print(f"\nProcessing column for kurtosis calculation: {col}")

            kurtosis_value = kurtosis(numeric[col].dropna())  # Dropping NaN to avoid calculation issues
            print(f"Kurtosis of '{col}': {kurtosis_value}")

            if kurtosis_value > 0:
                print(f"'{col}' is Leptokurtic (heavy tails of outliers).")
            elif kurtosis_value == 0:
                print(f"'{col}' is Mesokurtic (normal distribution).")
            else:
                print(f"'{col}' is Platykurtic (light tails of outliers).")
        
        print("Total no. of elements: ",size)
        print("*"*100)
        print("Shape of the dataset: ",shape)
        print("*"*100)
        print("Dimentions of the dataset: ",dimentions)
        print("*"*100)
        print("Information of the dataset: ",info)
        print("*"*100)
        print("Statistical summary of the dataset: ",summary)
        print("*"*100)
        print("Name of the columns: ",columns)
        print("*"*100)
        print("Total no. of columns: ",n_col)
        print("*"*100)
        print("Datatypes of the dataset: ",types)
        print("*"*100)
        print("No. of duplicated values: ",duplicated)
        print("*"*100)
        print("Percentage (%) of duplicated values: ",p_dup*100)
        print("*"*100)
        print("No. of missing values: ",missing)
        print("*"*100)
        print("Percentage (%) of missing values: ",p_miss*100)
        print("*"*100)
        print("Numerical columns are: ",numeric)
        print("*"*100)
        print("No. of numrerical columns are: ",n_num)
        print("*"*100)
        print("Categoical cloumns are: ",categoical)
        print("*"*100)
        print("No. of categoical columns are: ",n_col)
        print("*"*100)
        print("Date type columns are: ",datetime)
        print("*"*100)
        print("Numerical correlations: ",corr)
        print("*"*100)

class DataTypeHandler():

    def __init__(df):
        df = df

    def to_string(col):
        try:
            df[col] = df[col].astype(str)
            print(f"Converted column '{col}' to string.")
        except Exception as e:
            print(f"Error converting '{col}' to string: {e}")

    def to_int(col):
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            print(f"Converted column '{col}' to int.")
        except Exception as e:
            print(f"Error converting '{col}' to int: {e}")

    def to_float(col):
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            print(f"Converted column '{col}' to float.")
        except Exception as e:
            print(f"Error converting '{col}' to float: {e}")

    def to_datetime(col):
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted column '{col}' to datetime from int.")
        except Exception as e:
            print(f"Error converting '{col}' to datetime from int: {e}")

class MissingValueHandler():

    def __init__(df):
        df = df
    
    def mean(col):
        try: 
            if df[col].dtype in ['int64', 'float64']:
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
                print(f"Filled missing values in '{col}' with mean: {mean_value}")
            else:
                print(f"Cannot fill for column '{col}' as it is non-numeric")

        except Exception as e:
            print(f"Cannot fill mean for column '{col}' due to '{e}'.")

    def median(col):
        try: 
            if df[col].dtype in ['int64', 'float64']:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in '{col}' with mean: {median_value}")
            else:
                print(f"Cannot fill for column '{col}' as it is non-numeric")

        except Exception as e:
            print(f"Cannot fill median for column '{col}' due to '{e}'.")

    def mode(col):
        try: 
            if df[col].dtype in ['int64', 'float64']:
                median_value = df[col].mode().iloc[0]
                df[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in '{col}' with mean: {median_value}")
            else:
                print(f"Cannot fill for column '{col}' as it is non-numeric")

        except Exception as e:
            print(f"Cannot fill mode for column '{col}' due to '{e}'.")

    def b_fill(col):
        try: 
            if df[col].dtype in ['int64', 'float64']:
                df[col].bfill(inplace=True)
                print(f"Filled missing values in '{col}' using backward fill.")
            else:
                print(f"Cannot fill for column '{col}' as it is non-numeric")

        except Exception as e:
            print(f"Cannot bfill for column '{col}' due to '{e}'.")

    def f_fill(col):
        try: 
            if df[col].dtype in ['int64', 'float64']:
                df[col].ffill(inplace=True)
                print(f"Filled missing values in '{col}' using forward fill.")
            else:
                print(f"Cannot fill for column '{col}' as it is non-numeric")

        except Exception as e:
            print(f"Cannot ffill for column '{col}' due to '{e}'.")

    def linear(col):
        try: 
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].interpolate(method='linear')
                print(f"Filled missing values in '{col}' using linear value.")
            else:
                print(f"Cannot fill for column '{col}' as it is non-numeric")

            
        except Exception as e:
            print(f"Cannot linear fill for column '{col}' due to '{e}'.")

    def polynomial(col):
        try: 
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].interpolate(method='polynomial',order=2)
                print(f"Filled missing values in '{col}' using polynomial value.")
            else:
                print(f"Cannot fill for column '{col}' as it is non-numeric")
                
        except Exception as e:
            print(f"Cannot polynomial fill for column '{col}' due to '{e}'.")

    def drop(col):
        try:
            initial_count = df.shape[0]
            df.dropna(subset=[col], inplace=True)
            final_count = df.shape[0]
            print(f"Dropped rows with missing values in '{col}'. Rows removed: {initial_count - final_count}")
        except Exception as e:
            print(f"Cannot drop/delete values for column '{col}' due to '{e}'.")


class OutlierHander():

    def __init__(df):
        df = df
    
    def iqr_capping(col):
        try:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap the values outside the bounds
                df[col] = np.where(
                    df[col] < lower_bound, lower_bound,
                    np.where(df[col] > upper_bound, upper_bound, df[col])
                )

                print(f"Outliers in '{col}' have been capped using the IQR method.")
                print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
            else:
                print(f"Cannot apply IQR capping to non-numeric column '{col}'.")

        except Exception as e:
            print("Cannot perform IQR outlier capping to the column '{col}' due to {e}")
    

    def zscore_capping(col):
        try:
            if df[col].dtype in ['int64', 'float64']:
                threshold = 3
                mean = df[col].mean()
                std = df[col].std()
                z_scores = stats.zscore(df[col].dropna())

                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

                # Cap the values outside the bounds based on Z-score threshold
                df[col] = np.where(
                    z_scores < -threshold, lower_bound,
                    np.where(z_scores > threshold, upper_bound, df[col])
                )

                print(f"Outliers in '{col}' have been capped using the Z-score method with a threshold of {col}.")
                print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
            else:
                print(f"Cannot apply Z-score capping to non-numeric column '{col}'.")
        except Exception as e:
            print("Cannot perform Z-Score outlier capping to the column '{col}' due to {e}")


class NumericalScaler():

    def __init__(df):
        df = df
    
    def standardscaler(col):
        try:
            if df[col].dtype in ['int64', 'float64']:
                scaler = StandardScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
                print(f"'{col}' has been scaled using Standard Scaler.")
            else:
                print(f"Cannot apply Standard Scaler to non-numeric column '{col}'.")
        except Exception as e:
            print(f"Cannot apply Standard Scaler to column '{col}' due to '{e}'")


    def robustscaler(col):
        try:
            if df[col].dtype in ['int64', 'float64']:
                scaler = RobustScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
                print(f"'{col}' has been scaled using Robust Scaler.")
            else:
                print(f"Cannot apply Robust Scaler to non-numeric column '{col}'.")
        except Exception as e:
            print(f"Cannot apply Robust Scaler to column '{col}' due to '{e}'")

class VariableTransformer():

    def __init__(df):
        df = df

    def binner(col,bins):
        """
        Perform binning on a numerical column.
        
        Parameters:
        col (str): The name of the column to be binned.
        bins (list): The boundaries for the bins.
        labels (list, optional): The labels for each bin.
        """
        try:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = pd.cut(df[col], bins=bins, include_lowest=True)
                print(f"'{col}' has been binned into {len(bins)-1} categories.")
            else:
                print(f"Binning can only be applied to numeric columns. '{col}' is not numeric.")
        except Exception as e:
            print(f"Cannot apply Binning to column '{col}' due to '{e}'")

    def log_transformer(col):
        """
        Apply log transformation to a numerical column.
        
        Parameters:
        col (str): The name of the column to be log-transformed.
        """
        try:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = np.log1p(df[col])  # log1p to handle log(0)
                print(f"Log transformation applied to '{col}'.")
            else:
                print(f"Log transformation can only be applied to numeric columns. '{col}' is not numeric.")
        except Exception as e:
            print(f"Cannot apply Log transformation to column '{col}' due to '{e}'")

    def sqrt_transformer(col):
        """
        Apply square root transformation to a numerical column.
        
        Parameters:
        col (str): The name of the column to be square root transformed.
        """
        try:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = np.sqrt(df[col])
                print(f"Square root transformation applied to '{col}'.")
            else:
                print(f"Square root transformation can only be applied to numeric columns. '{col}' is not numeric.")
        except Exception as e:
            print(f"Cannot apply Square root transformation to column '{col}' due to '{e}'")
    def label_encoding(col):
        """
        Apply label encoding to a categorical column.
        
        Parameters:
        col (str): The name of the column to be label encoded.
        """
        try:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                print(f"Label encoding applied to '{col}'.")
            else:
                print(f"Label encoding can only be applied to categorical columns. '{col}' is not categorical.")
        except Exception as e:
            print(f"Cannot apply Label encoding to column '{col}' due to '{e}'")
    def one_hot_encoding(col):
        """
        Apply one-hot encoding to a categorical column, modifying the original column without adding new ones.
        
        Parameters:
        col (str): The name of the column to be one-hot encoded.
        """
        try:
            if df[col].dtype == 'object':
                one_hot_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
                df[col] = one_hot_encoder.fit_transform(df[[col]])
                print(f"One-hot encoding applied to '{col}'.")
            else:
                print(f"One-hot encoding can only be applied to categorical columns. '{col}' is not categorical.")
        except Exception as e:
            print(f"Cannot apply One-hot encoding to column '{col}' due to '{e}'")