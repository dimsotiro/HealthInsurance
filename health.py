#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("C:\\Users\\dimso\\Downloads\\insurance.csv")


# In[5]:


# Round the 'charges' column to 2 decimal places
df['charges'] = df['charges'].round(2)

print(df)


# In[7]:


df['sex']=df['sex'].replace({"female":1, "male":0})
df['smoker']=df['smoker'].replace({"yes":1, "no":0})
print(df)


# In[9]:


df.info()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Define numeric columns for visualization (update if needed)
numeric_columns = ['age', 'bmi', 'children', 'charges']

# Create a figure with subplots for box plots
fig, axes = plt.subplots(nrows=1, ncols=len(numeric_columns), figsize=(15, 5))

# Loop through numeric columns to create box plots for original data
for i, column in enumerate(numeric_columns):
    sns.boxplot(ax=axes[i], x=df[column])
    axes[i].set_title(f'Box Plot of {column}')

plt.tight_layout()
plt.show()

# Create a figure with subplots for histograms
plt.figure(figsize=(15, 10))

# Plot histograms for the original DataFrame
for i, column in enumerate(numeric_columns):
    plt.subplot(2, len(numeric_columns), i + 1)
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f'Histogram of {column}')

plt.tight_layout()
plt.show()


# In[12]:


import scipy.stats as stats

# Q-Q Plot for charges
plt.figure(figsize=(12, 6))
stats.probplot(df['charges'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Charges', fontsize=20)
plt.grid(True)
plt.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Box Plot for 'sex' vs 'charges'
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='charges', data=df, palette='pastel', hue='sex', legend=False)
plt.title('Box Plot of Charges by Sex')
plt.show()

# Box Plot for 'smoker' vs 'charges'
plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='charges', data=df, palette='pastel', hue='smoker', legend=False)
plt.title('Box Plot of Charges by Smoker Status')
plt.show()

# Box Plot for 'region' vs 'charges'
plt.figure(figsize=(10, 6))
sns.boxplot(x='region', y='charges', data=df, palette='pastel', hue='region', legend=False)
plt.title('Box Plot of Charges by Region')
plt.show()


# In[17]:


# Kolmogorov-Smirnov Test
ks_test = stats.kstest(df['charges'], 'norm', args=(df['charges'].mean(), df['charges'].std()))
print('Kolmogorov-Smirnov Test Results:')
print('Statistic:', ks_test.statistic)
print('p-value:', ks_test.pvalue)


# In[19]:


# Perform Mann-Whitney U test for charges between males and females
smoker_result = stats.mannwhitneyu(df[df['smoker'] == 1]['charges'], df[df['smoker'] == 0]['charges'])

print('Mann-Whitney U Test (Charges vs Smoker):')
print('U-statistic:', smoker_result.statistic)
print('p-value:', smoker_result.pvalue)
print(
    "There is significant difference in the distribution of charges across the smokers and non-smokers"
)


# In[21]:


# Perform Mann-Whitney U test for charges between males and females
sex_result = stats.mannwhitneyu(df[df['sex'] == 1]['charges'], df[df['sex'] == 0]['charges'])

print('Mann-Whitney U Test (Charges vs Sex):')
print('U-statistic:', sex_result.statistic)
print('p-value:', sex_result.pvalue)
print(
    "There is no significant difference in the distribution of charges across the sex"
)


# In[23]:


# Perform Kruskal-Wallis test
kruskal_result = stats.kruskal(
    df[df['region'] == 'northwest']['charges'],
    df[df['region'] == 'southwest']['charges'],
    df[df['region'] == 'northeast']['charges'],
    df[df['region'] == 'southeast']['charges']
)

print('Kruskal-Wallis Results:')
print('H-statistic:', kruskal_result.statistic)
print('p-value:', kruskal_result.pvalue)
print(
    "There is no significant difference in the distribution of charges across the four regions (northwest, southwest, northeast, southeast) at the typical significance level of 0.05"
)


# In[25]:


df_encoded = pd.get_dummies(df, columns=['region'], drop_first=True)
print(df_encoded)


# In[27]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Splitting the dataset
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regression
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

# Evaluating
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing results
print(f"Random Forest Results:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


# In[29]:


# Example: 40-year-old male, BMI 30, 2 children, non-smoker, living in the northwest region
new_data = {
    'age': [40],
    'sex': [0],  # 0 for male
    'bmi': [30],
    'children': [2],
    'smoker': [0],  # 0 for non-smoker
    # One-hot encoding for regions: Setting the region northwest to 1 and others to 0
    'region_northwest': [1],
    'region_southeast': [0],
    'region_southwest': [0]
}

# Convert new data to a DataFrame
X_new = pd.DataFrame(new_data)

# Predict using the trained model
y_new_pred = model.predict(X_new)

# Output the predictions
print("Predicted Charges for new data:", y_new_pred)


# In[151]:


import numpy as np
import pandas as pd

# Number of samples to generate
n_samples = 10

# Generate random data for the features
random_data = {
    'age': np.random.randint(18, 65, size=n_samples),        
    'sex': np.random.randint(0, 2, size=n_samples),          
    'bmi': np.random.uniform(15, 40, size=n_samples),        
    'children': np.random.randint(0, 5, size=n_samples),    
    'smoker': np.random.randint(0, 2, size=n_samples),       
}

# Convert random data to a DataFrame
X_random = pd.DataFrame(random_data)

# One-hot encoding for regions: Randomly assign one region to be '1'
regions = ['region_northwest', 'region_southeast', 'region_southwest']
X_random[regions] = 0  # Initialize all region columns to 0

# Randomly choose a region for each sample to set as 1
X_random.loc[:, 'region'] = np.random.choice(regions, size=n_samples)  # Randomly select a region
for region in regions:
    X_random[region] = (X_random['region'] == region).astype(int)  # Set selected region to 1, others to 0

# Drop the temporary 'region' column used for selection
X_random = X_random.drop(columns=['region'])

# Predict using the trained model
y_random_pred = model.predict(X_random)

# Print predictions
print("Randomly Generated Data:")
print(X_random)
print("\nPredicted Charges for Random Samples:")
print(y_random_pred)

# Example: Using the mean charges from the dataset as a proxy for the true value
y_true = np.full(n_samples, df_encoded['charges'].mean())

# Calculate evaluation metrics
mae = mean_absolute_error(y_true, y_random_pred)
mse = mean_squared_error(y_true, y_random_pred)

print("\nEvaluation Metrics for Random Predictions:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)


# In[ ]:





# In[ ]:





# In[ ]:




