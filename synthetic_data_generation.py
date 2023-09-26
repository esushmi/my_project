#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Install DataSynthesizer package
get_ipython().system('pip install DataSynthesizer')


# In[29]:


# Import necessary libraries and modules
import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network


# In[65]:


# Import necessary libraries and modules
import boto3
import sagemaker
from sagemaker.session import Session
import datetime
import numpy as np


# <font color="yellow">use the commented block if the input file has to be read from s3 bucket

# In[4]:


# #specify s3 bucket info
# s3_bucket ="sagemaker-eu-west-1-610914939903"
# s3_prefix ="dynamic_campaign_dataset"

# sagemaker_session = sagemaker.Session()
# region = sagemaker_session.boto_region_name
# role = sagemaker.get_execution_role()


# In[5]:


# s3 = boto3.client('s3')
# obj = s3.get_object(
#          Bucket = 'sagemaker-eu-west-1-610914939903', 
#          Key = 'dynamic_campaign_dataset/prod_disp_one_to_one.csv'
#       )
# data = pd.read_csv(obj['Body'])


# In[31]:


#read the input file
data = pd.read_csv("spend_50_500.csv")


# <font color="yellow">preprocessing

# In[32]:


data.columns


# In[33]:


len(data.columns)
data.shape


# In[35]:


#reatin necessary columns
data = data[['campaign_name','ad_group_name','asin', 'status',  'spend','date',
       'targeting_type', 'impressions',
        'clicks','spend',
        'cost_per_click_cpc',
        '14_day_total_units','total_return_on_advertising_spend_roas','28days_avg_demand']]


# In[36]:


data_backup = data.copy()


# In[37]:


data_backup['status'].value_counts()


# In[38]:


data_backup = data.copy()


# In[39]:


# Feature extraction: Convert date to datetime and derive year, weekday, 
#and month features, then drop the original 'date' column
data['date'] = pd.to_datetime(data['date'], utc=True)
data['year'] = data.date.dt.year
data['weekday'] = data.date.dt.dayofweek
data['day'] = data.date.dt.day
data['month'] = data.date.dt.month

# Dropping the date column
data.drop('date', axis=1, inplace=True)


# In[40]:


data.info()


# In[41]:


# Iterate through columns with object data type, display value counts for each column
for col in data.select_dtypes(include=['object']).columns:
    print(f'{col} value count')
    counts = data[col].value_counts()
    print(counts)


# In[42]:


#check the number of campaings
print(len(data['campaign_name'].unique()))


# In[43]:


#convert the processed dataframe to csv file
data = data.to_csv("synthetic_FSN_data_spend.csv")


# In[44]:


input_data ="synthetic_FSN_data_spend.csv"
synthetic_data = 'dcm_synthetic_spend.csv'


# <font color="yellow">synthetic data generation

# In[45]:


# Setting the mode to correlated_attribute_mode during synthetic data generation.
# This mode ensures that the synthetic data will preserve the correlations between attributes 
# found in the original dataset,
# making the generated data realistic and suitable for maintaining attribute relationships.

mode = 'correlated_attribute_mode'


# In[46]:


# Generate the description file name based on the correlated_attribute_mode
description_file = f'{mode}description_onw_to_one.json'
print(description_file)


# In[47]:


# Define the parameters for synthetic data generation

# An attribute is categorical if its number of unique values are less than this threshold
threshold_value = 11

# specify categorical attributes
categorical_attributes = {'status': True, 'targeting_type': True, 'campaign_name': True
                         ,'ad_group_name': True,'asin': True}

# # specify candidate keys
# candidate_keys = {'campaign_name': True}

# Tune noise level. Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy
epsilon = 0.0

# The maximum number of parents in Bayesian network
degree_of_bayesian_network = 3

# Number of tuples generated in synthetic dataset.
num_tuples_to_generate = 2000


# In[48]:


# Generate and save dataset description using DataDescriber in correlated attribute mode.
# Constructs Bayesian Network (BN)
describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=categorical_attributes
                                                        )
describer.save_dataset_description_to_file(description_file)


# In[49]:


display_bayesian_network(describer.bayesian_network)


# In[50]:


# Generate synthetic data based on the dataset description and 
# save it to the specified file(synthetic_data = 'dcm_synthetic_spend.csv').

generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
generator.save_synthetic_data(synthetic_data)


# <font color="yellow">plotting the graphs to check if synthetic data attributes are close to original data

# In[85]:


df_syn = pd.read_csv("dcm_synthetic_cluster1.csv")
df_org = pd.read_csv("cluster1.csv")


merged_df = pd.merge(df_org, df_syn, on=['year',	'weekday',	'day',	'month'], how='inner')


# In[27]:



import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(20, 6))

# Plot budget_x and budget_y as a line chart
plt.plot(merged_df['total_return_on_advertising_spend_roas_y'], label='clicks_y')
plt.plot(merged_df['total_return_on_advertising_spend_roas_x'], label='clicks_x')


# Set the axis labels and title
plt.xlabel('Index')
plt.ylabel('total_return_on_advertising_spend_roas')
plt.title('roas Comparison')

# Show the legend
plt.legend()

# Show the plot
plt.show()


# In[50]:



import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(20, 6))

# Plot spend_x and spend_y as a line chart
plt.plot(merged_df['spend_y'], label='spend_y')
plt.plot(merged_df['spend_x'], label='spend_x')


# Set the axis labels and title
plt.xlabel('Index')
plt.ylabel('spend')
plt.title('spend Comparison')

# Show the legend
plt.legend()

# Show the plot
plt.show()


# In[66]:


original_df = pd.read_csv(input_data)
synthetic_df = pd.read_csv(synthetic_data)


# In[68]:


attribute_description = read_json_file(description_file)['attribute_description']

inspector = ModelInspector(original_df, synthetic_df, attribute_description)

for attribute in ['budget',	'impressions',	'clicks',	'14_day_total_orders',	'cost_per_click_cpc','total_return_on_advertising_spend_roas']:
    inspector.compare_histograms(attribute)


# <font color="yellow">process the synthetic data to get the final columns 

# In[61]:


syn = pd.read_csv("dcm_synthetic_spend.csv")
syn.shape


# In[62]:


# get date columns 
syn['date'] = pd.to_datetime(syn[['day', 'month', 'year']], errors='coerce')


# In[63]:


#drop weekday, day, month year columns
columns_to_drop = ['weekday', 'day', 'month', 'year']
syn = syn.drop(columns_to_drop, axis=1)


# In[66]:


#calculate ctr(as we exclude mathematically generated columns while generating synthetic data)
syn['click_thru_rate_ctr'] = np.where(syn['impressions'] != 0, (syn['clicks'] / syn['impressions']) * 100, 0)


# In[67]:


#get the final csv with the all the columns
syn.to_csv("synthetic_spend.csv")

