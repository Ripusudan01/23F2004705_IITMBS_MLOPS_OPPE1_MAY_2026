#!/usr/bin/env python
# coding: utf-8

# # Data Processing Reference for a Single Stock - AARTIIND
# 
# This notebook processes stock price data and creates the following features:
# - `rolling_avg_10`: 10-minute rolling average of close price
# - `volume_sum_10`: Total volume traded over last 10 minutes
# - `target`: Binary indicator if stock goes up (1) or down (0) in next 5 minutes

# In[101]:


import pandas as pd
import numpy as np


# ## 1. Load Data

# In[102]:


# Load data from file (for demonstration)
df = pd.read_csv("./AARTIIND__EQ__NSE__NSE__MINUTE.csv")

# For loading from file, use:
# df = pd.read_csv('your_data.csv')

print("Data loaded successfully!")
print(f"Shape: {df.shape}")
df.head()


# ## 2. Data Preprocessing

# In[103]:


# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['stock_name'] = "AARTIIND"

# Sort by timestamp (important for time series operations)
df = df.sort_values('timestamp').reset_index(drop=True)

# Set timestamp as index for easier time-based operations
df.set_index('timestamp', inplace=True)

print("Data preprocessed!")
print(f"Date range: {df.index.min()} to {df.index.max()}")
df.head()


# In[104]:


# Fill missing values with prior values (forward fill)
df.ffill(inplace=True)

# Check for missing values after filling
print("Missing values after filling:")
print(df.isnull().sum())


# ## 3. Feature Engineering

# In[105]:


# Feature 1: 10-minute rolling average of close price
# Using window='10T' for 10 minutes
df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()

# Feature 2: Total volume traded over last 10 minutes
df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()

print("Rolling features created!")
df[['close', 'volume', 'rolling_avg_10', 'volume_sum_10']].head(5)


# In[106]:


# Remove rows where rolling_avg_10 or volume_sum_10 is NaN
df.dropna(subset=['rolling_avg_10', 'volume_sum_10'], inplace=True)

print("Removed rows with NaN in rolling_avg_10 or volume_sum_10.")
print(f"New shape: {df.shape}")


# In[107]:


# Feature 3: Target variable - does stock go up in next 5 minutes?
# Shift close price backwards by 5 minutes to get future price
df['close_5min_future'] = df['close'].shift(-5)

# Create binary target: 1 if price goes up, 0 if it goes down or stays same
df['target'] = (df['close_5min_future'] > df['close']).astype(int)

print("Target variable created!")
print(f"\nTarget distribution:")
print(df['target'].value_counts())
print(f"\nTarget ratio (up/total): {df['target'].mean():.2%}")


# ## 5. Final Dataset

# In[108]:


# Drop close_5min_future as its a signal to target variable
df.drop(["close_5min_future"],axis=1,inplace=True)

# Drop rows where target is NaN
df_clean = df.dropna(subset=['target']).copy()

# Take the last 20 observations for the test set
test_df = df_clean.tail(20).copy()

# Remove the test observations from the training set
df_clean = df_clean.iloc[:-20].copy()


print(f"\nTest dataset (last 20 observations):")
print(f"Shape: {test_df.shape}")
print(f"\nClean dataset (removed last 20 observations for test):")
print(f"Shape: {df_clean.shape}")


# ## 7. Export Processed Data

# In[109]:


# Save processed data to CSV
output_file = 'train.csv'
df_clean.to_csv(output_file)
print(f"Processed data saved to: {output_file}")

# Save the test dataset
test_output_file = 'test.csv'
test_df.to_csv(test_output_file)
print(f"Test data saved to: {test_output_file}")

# Display info about the saved files
print(f"\nColumns in processed output file: {list(df_clean.columns)}")
print(f"Total rows in processed data: {len(df_clean)}")
print(f"\nColumns in test output file: {list(test_df.columns)}")
print(f"Total rows in test data: {len(test_df)}")


# In[100]:




