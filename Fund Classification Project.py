#!/usr/bin/env python
# coding: utf-8

# # Fund Classification Project

# ## Import Libraries

# In[165]:


# Import necessary libraries for data manipulation and machine learning
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# ## Load Data

# In[166]:


data_path = 'AC6030_Assignment3_Data.xlsx'
data = pd.read_excel(data_path, sheet_name='Fund_Performance')


# In[167]:


# Load labels
labels_df = pd.read_excel(data_path, sheet_name='Labels')
labels_df.columns = ['Fund', 'Type']


# # Data Inspection

# In[168]:


# Display basic information about the dataset
print("Data Overview:")
print(data.head())
print("\nData Info:")
data.info()


# In[169]:


# Summary statistics for the dataset to inspect central tendency and dispersion
print("\nData Description:")
print(data.describe())


# In[170]:


# Check for missing values in the dataset
print("\nMissing Values Check:")
print(data.isnull().sum())


# ## Feature Extraction

# In[171]:


# Step 2: Feature Extraction
# Define a function to calculate the linear trend of time series data
def get_trend_feature(y):
    X = np.arange(len(y)).reshape(-1, 1)  # Time index as independent variable
    model = LinearRegression().fit(X, y)
    return model.coef_[0]  # Return the slope of the regression line


# In[172]:


# Initialize a DataFrame to store calculated features
features_df = pd.DataFrame()


# In[173]:


# Calculate descriptive statistics as features
features_df['mean'] = data.iloc[:, 1:].apply(np.mean, axis=0)
features_df['std'] = data.iloc[:, 1:].apply(np.std, axis=0)
features_df['skew'] = data.iloc[:, 1:].apply(skew, axis=0)
features_df['kurtosis'] = data.iloc[:, 1:].apply(kurtosis, axis=0)


# In[174]:


# Calculate the trend for each fund
features_df['trend'] = data.iloc[:, 1:].apply(get_trend_feature, axis=0)


# In[175]:


# Calculate volatility using a 12-month rolling standard deviation
rolling_volatility = data.iloc[:, 1:].rolling(window=12, min_periods=1).std()
features_df['volatility_12m'] = rolling_volatility.apply(np.mean, axis=0)


# In[176]:


# Calculate momentum as the change between the first and last recorded return
features_df['momentum'] = data.iloc[-1, 1:] - data.iloc[0, 1:]


# In[177]:


# Merge the features with the labels
full_data_df = features_df.merge(labels_df, left_index=True, right_on='Fund')
full_data_df.set_index('Fund', inplace=True)


# In[178]:


# Display the features DataFrame
print("Extracted Features for Each Fund:")
print(features_df.head())
print("\nStatistical Summary of Features:")
print(features_df.describe())


# In[179]:


# Optionally, visualize some of the features
plt.figure(figsize=(10, 6))
plt.subplot(121)
features_df['mean'].hist()
plt.title('Distribution of Mean Returns')
plt.subplot(122)
features_df['std'].hist()
plt.title('Distribution of Standard Deviation of Returns')
plt.show()


# ## Prepare Data for Modeling

# In[180]:


# Prepare data for modeling
X = full_data_df.drop(columns=['Type'])
y = full_data_df['Type']


# In[181]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[182]:


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[210]:


# Initialize and train classification models
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)


# In[209]:


log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)


# ## Train and Evaluate Models

# In[200]:


# Step 4: Model Training and Evaluation
# Initialize the machine learning models
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)


# In[208]:


# Train the models on the training data
log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)


# In[207]:


# Make predictions on the testing data
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_svm = svm.predict(X_test_scaled)


# In[206]:


# Print the accuracy and classification report for each model
print("Accuracy of Logistic Regression: ", accuracy_score(y_test, y_pred_log_reg))
print("Accuracy of Random Forest: ", accuracy_score(y_test, y_pred_rf))
print("Accuracy of SVM: ", accuracy_score(y_test, y_pred_svm))
print("Classification Report for Logistic Regression:\n", classification_report(y_test, y_pred_log_reg))
print("Classification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))
print("Classification Report for SVM:\n", classification_report(y_test, y_pred_svm))


# In[ ]:





# In[ ]:




