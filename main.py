# -*- coding: utf-8 -*-

##############################
### IMPORTING PYTHON PKGS  ###
##############################
import pandas as pd
import numpy as np
#import pandas_profiling as pp
from ipywidgets import widgets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from date_cleaner import date_cleaner
from charset_cleaner import charset_cleaner
from server_cleaner import server_cleaner 
from state_cleaner import state_cleaner
from plot_confusion_metrics import plot_confusion_matrix_and_print_metrics
from train_and_evaluate import train_and_evaluate_model
import pickle


# %matplotlib inline

#pd.set_option('max_columns', None)
#pd.set_option('max_rows', None)

## DATASET INGESTION ##
## Read the CSV file in the data frame below and print the first 5 rows

df = pd.read_csv("/content/drive/MyDrive/dataset.csv")
df.head()

#############################################
### EXPLORATORY DATA ANALYSIS STARTS HERE ###
#############################################

df.drop("URL", axis=1, inplace=True)

## Handling Data Types of Features and Missing Values STARTS HERE##

# General Information related to features/columns
df.info()

## Usin describe method on the dataset to generate basic statistics related to each column ###
df.describe()

## Vsually understanding the Dataset
df['Type'].value_counts().plot.pie(autopct='%.2f')
plt.title("The Distribution of URLs in the Dataset: Malicious vs Benign")
plt.show()

## Boxplots graphs to understand the distribution of a certain column
plt.figure(figsize=(12,6))
sns.boxplot(data=df,x='Type',y='NUMBER_SPECIAL_CHARACTERS');

plt.figure(figsize=(12,6))
sns.boxplot(data=df,x='Type',y='URL_LENGTH');

## Using pandas-profiling tool for EDA. It takes dataframe and generates a report with statistics and graphs
#pp.ProfileReport(df)

### PREPROCESSING STARTS HERE ##
# It includes
  #Removal of duplicates
  #Preprocessing of date features such as converting them into datetime variable from string values.
  #Imputation of missing values. They should be replaced with some number or dropped.
  #Scaling of numerical values between 0 and 1.
  #Encoding of categorical features before feeding them into a model.

# Copy df into df_preprocessed variable here
df_preprocessed = df.copy()
df_preprocessed.shape

# Removing duplicates
df_preprocessed = df_preprocessed[~df_preprocessed.duplicated()].copy()
df_preprocessed.shape

df_preprocessed.head()




df_preprocessed.WHOIS_REGDATE = df_preprocessed.WHOIS_REGDATE.apply(date_cleaner)
df_preprocessed["WHOIS_REGDATE"] = pd.to_datetime(df_preprocessed.WHOIS_REGDATE, format="%d/%m/%Y", errors="coerce")
df_preprocessed["WHOIS_REGDATE"].value_counts(dropna=False).sort_index().head(10)

df_preprocessed.WHOIS_UPDATED_DATE = df_preprocessed.WHOIS_UPDATED_DATE.apply(date_cleaner)
df_preprocessed["WHOIS_UPDATED_DATE"] = pd.to_datetime(df_preprocessed.WHOIS_UPDATED_DATE, format="%d/%m/%Y", errors="coerce")
df_preprocessed["WHOIS_UPDATED_DATE"].value_counts(dropna=False).sort_index().head(10)

## #Handling Missing Values

df_preprocessed.isnull().sum()

# Impute with median as median values are more robust to mean values in case of outliers.
df_preprocessed.loc[:, "IS_CONTENT_LENGTH_IMPUTED"] = df_preprocessed.CONTENT_LENGTH.isna()
df_preprocessed.loc[df_preprocessed.CONTENT_LENGTH.isna(), "CONTENT_LENGTH"] = df_preprocessed.CONTENT_LENGTH.median()

df_preprocessed.loc[:, "IS_WHOIS_REGDATE_IMPUTED"] = df_preprocessed.WHOIS_REGDATE.isna()
df_preprocessed.loc[df_preprocessed.WHOIS_REGDATE.isna(), "WHOIS_REGDATE"] = df_preprocessed.WHOIS_REGDATE.median()

df_preprocessed.loc[:, "IS_WHOIS_UPDATED_DATE_IMPUTED"] = df_preprocessed.WHOIS_UPDATED_DATE.isna()
df_preprocessed.loc[df_preprocessed.WHOIS_UPDATED_DATE.isna(), "WHOIS_UPDATED_DATE"] = df_preprocessed.WHOIS_UPDATED_DATE.median()


# Drop two instances instead of introducing additional features
df_preprocessed = df_preprocessed[~df_preprocessed.SERVER.isna()]
df_preprocessed = df_preprocessed[~df_preprocessed.DNS_QUERY_TIMES.isna()]

#Checking again for missing values in dataset
df_preprocessed.isnull().sum()

#Standardize/Scale Numerical Values
columns_to_scale  = ["CONTENT_LENGTH", "WHOIS_REGDATE", "WHOIS_UPDATED_DATE", "DNS_QUERY_TIMES",
                     "URL_LENGTH", "NUMBER_SPECIAL_CHARACTERS", "TCP_CONVERSATION_EXCHANGE",
                     "DIST_REMOTE_TCP_PORT", "REMOTE_IPS", "APP_BYTES", "SOURCE_APP_PACKETS",
                     "REMOTE_APP_PACKETS", "SOURCE_APP_BYTES", "REMOTE_APP_BYTES",
                     "APP_PACKETS"]

df_preprocessed.WHOIS_REGDATE = df_preprocessed.WHOIS_REGDATE.view("int64")
df_preprocessed.WHOIS_UPDATED_DATE = df_preprocessed.WHOIS_UPDATED_DATE.view("int64")

sc = StandardScaler()
scaled_columns = sc.fit_transform(df_preprocessed[columns_to_scale])
df_preprocessed[columns_to_scale] = scaled_columns
df_preprocessed[columns_to_scale].head()

#Encoding Categorical Variables
df_preprocessed.CHARSET.value_counts()

df_preprocessed.CHARSET = df_preprocessed.CHARSET.apply(charset_cleaner)
df_preprocessed.CHARSET.value_counts()

df_preprocessed.SERVER.value_counts()

common_servers = df_preprocessed.SERVER.value_counts()[df_preprocessed.SERVER.value_counts() > 10].index.to_list()


df_preprocessed.SERVER = df_preprocessed.SERVER.apply(server_cleaner)
df_preprocessed.SERVER.value_counts()

df_preprocessed.WHOIS_COUNTRY.value_counts()

common_countries = ["US", "CA", "ES", "AU", "PA", "GB", "JP", "CN", "IN", "UK"]
def country_cleaner(country):
    country = "GB" if country in ["[u'GB'; u'UK']", "UK"] else country
    return "other" if country not in common_countries else country

df_preprocessed.WHOIS_COUNTRY = df_preprocessed.WHOIS_COUNTRY.apply(country_cleaner)
df_preprocessed.WHOIS_COUNTRY.value_counts()

df_preprocessed.WHOIS_STATEPRO.value_counts()

common_states = df_preprocessed.WHOIS_STATEPRO.value_counts()[df_preprocessed.WHOIS_STATEPRO.value_counts() > 10].index.to_list()
common_states.pop(0)  # remove "None" from the common states
common_states

## clean WHOIS_STATEPRO column and check the cleaned values

df_preprocessed.WHOIS_STATEPRO = df_preprocessed.WHOIS_STATEPRO.apply(state_cleaner)

categorical_columns = ["CHARSET", "SERVER", "WHOIS_COUNTRY", "WHOIS_STATEPRO"]

for col in categorical_columns:
    encoded_columns = pd.get_dummies(df_preprocessed[col])
    for encoded_col in encoded_columns.columns:
        df_preprocessed[f"{col}_{encoded_col}"] = encoded_columns[encoded_col]
    df_preprocessed.drop(columns=[col], inplace=True)

df_preprocessed.reset_index(drop=True, inplace=True)
df_preprocessed.head()


## LEARNING FROM DATA ##

df_preprocessed.shape
# It says that total, I have 1724 instances with 75 features and 1 label

# Split labels from the features.
# Later, I wil split our dataset into train and test datasets

Y = df_preprocessed["Type"].to_numpy()
X = df_preprocessed.drop(columns=["Type"]).to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

## RANDOM FOREST ##
# train a Random Forest model to classify malicious urls
#After training the model, output confusion matrix and evaluation metrics such as f1-score, precision, and recall

model_params = {"n_estimators":100, "random_state":0, "class_weight": {0:1,1:8}}
random_forest_model = RandomForestClassifier(**model_params)

random_forest_model.fit(X_train, Y_train)

y_test_pred = random_forest_model.predict(X_test)

plot_confusion_matrix_and_print_metrics(y_test_pred, Y_test, model_params, "RANDOM_FOREST")

#Training models, one by one inefficient
#creating a dictionary that stores every model that we want to train with its parameters
#then implement a for loop to train all these models

##  Models Used
#1. NAIVE_BAYES
#2. KNN
#3. Decision Tree
#4. LOGISTIC_REGRESSION
#5. SVC
#6. RANDOM_FOREST
#7. LIGHTGBM
#8. NN

# Model parameters are explained in detail in the specified URLs for specific models below.
random_state = 0
class_weight = {0:1, 1:8}

# Load parameters from a pickle file
with open("MODEL_CLASSES_AND_PARAMETERS.pkl", "rb") as file:
    MODEL_CLASSES_AND_PARAMETERS = pickle.load(file)


for model_name in MODEL_CLASSES_AND_PARAMETERS:
    train_and_evaluate_model(model_name)
