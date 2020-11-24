#!/usr/bin/env python
# coding: utf-8

# #  QUESTION ONE (1)
# # DATA LOADING 

# In[1]:


# I would have required to use dask which is a parallel form of data loading if
# the size of the data were heaavier to increase time efficiciency and avoiding loading 
# all the data into the memory. An alternative is to chunk the data but it is not as efficient,comparatively
# because of the concatenation required at the end of the chunk process.
import pandas as pd
data=pd.read_csv('realestate_fraugster_case.csv',sep=';',index_col=False)
data.head(10)


# .

# #  QUESTION TWO (2)
# # DATA CLEANING 

# The data cleaning steps would be done in three phases as:
# 
# 
# #  PHASE 1: THE GENERAL OUTLOOK AND PROFILE OF THE DATASET

# # (a) Statistical Description
# The "describe" method of panda's dataframe gives the statistical description of the dataset.This helps to see the count of unique values,most frequent value,how the values deviate or vary from one another percentile, among others.
# 

# In[2]:


data.describe(include='all')


# In[3]:


data.info()


# # (b) Data Type Formats
# When trying to convert to specific datatypes, the rows that do not comply to the rules of this datatype are identified as errors.These would help in making suitable corrections on the identified observations.
# 
# 
# Also,possible operations on the columns depend on the datatype.The correct datatypes would also help to identify errors in the columns.In this section, emphasis would be made on the numeric columns while the non-numeric features would form the basis for the Inconsistency check in phase two

# The above information could help determine the need for type conversion
# The columns with 'object' datatypes need to be investigated to determine which ones would require conversion

# # i) The 'city','state', 'street' and 'type' object columns are non-numeric values
# The 'city','state', and 'type' look tempting to convert to the category dtypes for memory efficiency and optimization. 
# However, they would be left as object because the dataset is not large enough to cause memory issues.Also, if converted to category dtype, the addition of new distinct value into the columns would generate 'NaN' error. 

# # ii) The 'sale_date'  column being a date would be converted to date datatype. 
# 
# data['sale_date'] =  pd.to_datetime(data.sale_date, format='%Y-%m-%d %H:%M:%S')
# 

# data['sale_date'] =  pd.to_datetime(data.sale_date, format='%Y-%m-%d %H:%M:%S')
# 
# 
# Running the above line gives errors such as the one identified below
# 
# 
# ValueError: time data 1917-07-24 08:12:24% doesn't match format specified

# In[4]:


# The error causing rows were identified and corrected as follows
data["sale_date"].replace({"2013-12-19 04:05:22A": "2013-12-19 04:05:22", "1917-07-24 08:12:24%":"1917-07-24 08:12:24","1918-02-25 20:36:13&":"1918-02-25 20:36:13"}, inplace=True)


# # iii) The 'zip' and 'price' object columns have numeric values. These are supposed to be integer values.This is checked and the rows with errors are identified

# In[5]:


# The error causing rows were identified in the zip column
for j, value in enumerate(data['zip']):
   try:
      int(value)
   except ValueError:
      print('The identified error index {}: {!r}'.format(j, value))


# In[6]:


# The error causing rows were identified in the price column
for j, value in enumerate(data['price']):
   try:
      int(value)
   except ValueError:
      print('The identified error index {}: {!r}'.format(j, value))


# In[7]:


# The typographical error were corrected intuitively as follows
data["zip"].replace({"957f58": "95758"}, inplace=True)
data["price"].replace({"298000D": "298000"}, inplace=True)


# # iv) The 'longitude' and 'latitude' object columns have floating values. These are checked and the rows with errors identified

# In[8]:


for j, value in enumerate(data['longitude']):
   try:
      float(value)
   except ValueError:
      print('Index error for Longitude  {}: {!r}'.format(j, value))


# In[9]:


for j, value in enumerate(data['latitude']):
   try:
      float(value)
   except ValueError:
    print('Index error for Latitude  {}: {!r}'.format(j, value))


# In[10]:


# The typographical error were replaced intuitively as follows
data["longitude"].replace({"-121.2286RT": "-121.228678","-121.363757$": "-121.363757"}, inplace=True)
data["latitude"].replace({"38.410992C": "38.410992"}, inplace=True)


# In[11]:


#data = data.astype({'longitude': 'float64', 'latitude': 'float64','price':'int64','zip':'int64'})
data.info()


# #  PHASE 2: THE INCONSISTENCY CHECK

# (The data consistency check is used for the following:
# 
# Redundancy such as duplicates,irrelevant datapoints, format error among others in both the columns and the rows
# 
# 
# To do this, we check the consistency of non-numeric features (type, state, city and street) by:

# (i) Capitalization Consistency Check

# In[12]:


#The solution to the inconsistency in the case format (lower and upper cases) can be solved by either making all the letters 
# The upper case would be used in this case
data= data.apply(lambda x: x.astype(str).str.upper() if x.name in ['street', 'type','city','state'] else x)
data


# (ii) Duplicate Row Check

# In[13]:


# Duplicate row check would result into repetition with no new information in the dataset.
# Therefore, observations that have been earlier recorded should be deleted. It could happen as a result of double submission
# file merging, among others
data.drop_duplicates(inplace=True)
data


# (iii) Irrelevant/Redundant Row Check

# In[14]:


# Since, it is a real estate sales data. Some columns could be seen as unique identifiers. 
# Unavailability or missingness of this identifiers would render the observation(row) redundant
# An identifier here would be the Longitude and Lattitude. 
# This is because the house/bed/baths sold would not be identified without this information.
# Therefore, rows with this missing values should be removed
import numpy as np
data = data.dropna(axis=0, subset=['longitude','latitude'])
data


# (iv) Typographical and Format Errors

# The unique values of the non-numeric columns ('type','state','city', and 'street') as shown in Out[2]: above are free text, which is prone to typographical error and human discretion in its format used. A look at the unique values show these errors.
# 
# 
# As can be seen in the state column, there are typographical error as 'CA', 'CA3', 'CA-' is pointing to a singular state 'CA'.

# [1] The solution to the 'states' column can be either of:
# 
# 
# a) Delete the column since it is a single-valued column and would not help in any ML modelling task.
# 
# 
# b) Correct the spelling and typo-errors.
# 
# 
# For completeness of the dataset, I will just replace the values with 'CA'

# In[15]:


# Check the unique values in the 'state' column and also save a copy of the data with a new name
print(data.state.unique())
new_data=data.copy()


# In[16]:


#new_data.loc[new_data['state'] == 'CA']
new_data=data.loc[data['state'] == 'CA']
new_data.state.unique()


# [2] The solution to the 'type' column:

# In[31]:


#The unique values in the type column are replaced appropriately
new_data.type.unique()
new_data["type"].replace({"RESIDENTIAL%": "RESIDENTIAL","RESIDEN_TIAL": "RESIDENTIAL","RESIDENTIAL)": "RESIDENTIAL"}, inplace=True)
new_data.type.unique()


# [3] The solution to the 'city' column:

# In[18]:


# To check the count and unique values in the column
print(new_data.city.nunique())
new_data.city.unique()


# In[19]:


# One way to do this is to create a list of valid cities in California
# Then, check the "city" column with this list.
# Any value that is present in the 'city' column but not available in the actual
# city list would be investigated
actual_city=['SACRAMENTO', 'RANCHO CORDOVA', 'RIO LINDA', 'CITRUS HEIGHTS',
       'NORTH HIGHLANDS', 'ANTELOPE', 'ELK GROVE',
       'ELVERTA', 'GALT', 'CARMICHAEL', 'ORANGEVALE', 'FOLSOM',
       'ELK GROVE', 'MATHER', 'POLLOCK PINES', 'GOLD RIVER',
       'EL DORADO HILLS', 'RANCHO MURIETA', 'WILTON', 'GREENWOOD',
       'FAIR OAKS', 'CAMERON PARK', 'LINCOLN', 'PLACERVILLE',
       'MEADOW VISTA', 'ROSEVILLE', 'ROCKLIN', 'AUBURN', 'LOOMIS',
       'EL DORADO', 'PENRYN', 'GRANITE BAY', 'FORESTHILL',
       'DIAMOND SPRINGS', 'SHINGLE SPRINGS', 'COOL', 'WALNUT GROVE',
       'GARDEN VALLEY', 'SLOUGHHOUSE', 'WEST SACRAMENTO']
check_this= new_data[~new_data.city.isin(actual_city)].city
check_this


# In[20]:


#The unique values in the type column are replaced appropriately
new_data["city"].replace({"SACRAMENTO@": "SACRAMENTO","ELK GROVE<>": "ELK GROVE"}, inplace=True)
print(new_data.city.nunique())
new_data.city.unique()


# In[32]:


# Other possible typo-error that can be checked are whitespace, fullstop, among others
new_data['city'] = new_data['city'].str.strip() # delete whitespace.
new_data['city'] = new_data['city'].str.replace('\\.', '') # delete dot/full stop.
print(new_data.city.nunique())
new_data.city.unique()


# [4] The solution to the 'street' column:

# In[22]:


# To check the count of unique values in the column
new_data.street.nunique()


# In[33]:


# There is actually less to e done here because the unique values almost equal the number of observations
# Therefore,one way to clean the data is to emove blanks,dots,abbreviate some words, etc
new_data['street'] = new_data['street'].str.strip() # delete blankspaces
new_data['street'] = new_data['street'].str.replace('\\.', '') # delete dot/full stop.
print(new_data.street.nunique())


# In[24]:


#changing the datatypes after the corrections have been effected
datatype= {'price': int, 'zip': int,'longitude':float,'latitude':float}  
new_data = new_data.astype(datatype) 
new_data['sale_date'] =  pd.to_datetime(new_data.sale_date, format='%Y-%m-%d %H:%M:%S')
print(new_data.dtypes)


# #  PHASE 3: HANDLING THE MISSING VALUES

# In[25]:


new_data.isnull().values.any()


# There are no missing values in the refined data. However, there are 'zero' valued cells which could also mean that the missing values have been replaced with zero.If the zero values actually represent missing values. Then, there are a number of ways to handle this:
# 
# 
# (i) Single-Value Imputation(SI) which involves replacing the missing cells with a single value. It could be the mean,highest occuring values,among others.
# 
# 
# (ii) Multiple/Multivariate Imputation(MI) which involves the use of different values to replace the missing cell based on the distribution of the data. There are several state of the art methods to do this.
# 
# 
# My master thesis research was based on Classification with data irregularities (missing values and class imbalance).I implemented and compared different sota imputation algorithms such as Generative Adversarial Network (GAN) for building prediction. This could be a good alternatives to handling the missing values.
# The link to my thesis can be found here https://github.com/busarieedris/Classification-with-Data-Irregularities
# (There may be some restrictions on some data due to privacy concerns.It is a collaborative research with a foremost research institute in Germany)

# .

# #  QUESTION THREE (3)
# # DATA SAVING

# In[26]:


# Save the cleaned data with a better interactive name. This can be done with the '.to_csv' command
# But the instruction says 'write a new csv with a similar name with the cleaned data'.That is the reason for changing the cleaned data
# with a better name first.
clean_realestate_fraugster_case=new_data.copy()
clean_realestate_fraugster_case.to_csv('clean_realestate_fraugster_case.csv',index=False,sep=';')


# In[27]:


clean_realestate_fraugster_case.info()


# .

#  #                                     QUESTION FOUR (4)
# (A) what is the distance (in meters) between the cheapest sale and the most recent sale?
# 
# 
# SOLUTION / APPROACH
# 
# 
# To do this:
# 
# 
# STEP 1: You need the location (Longitude and Latitude) of the two points (The cheapest sale and the most recent sale).
# 
# 

# In[28]:


# LET X BE THE CHEAPEST SALE (i.e The least value in the 'price' column)  
lon_x=new_data.loc[new_data['price'].idxmin()]['longitude'] # The corresponding longitude for X 
lat_x=new_data.loc[new_data['price'].idxmin()]['latitude']  # The corresponding latitude for X


# In[29]:


# LET Y REPRESENT THE MOST RECENT SALE (i.e The most recent date in the 'sale_date' column)
lon_y=new_data.loc[new_data.sale_date.idxmax()]['longitude'] # The corresponding longitude for the most recent sale
lat_y=new_data.loc[new_data.sale_date.idxmax()]['latitude']  # The corresponding latitude for the most recent sale


# STEP 2: Calculate the difference in distance between these two points

# In order to get the distance between two coordinate points, there are quite some formulars for such calculations with varying degree of accuracy.
# 
# Some of the methods are:
# 
# 1) Haversine formula: It is used to determine the distance between two points based on the law of Haversine.
# 
# 
# 2) Vincenty Formula: It is a distance calculation based on the fact that the earth is oblate spherical.It has an accuracy of almost 1mm
# 
# 
# Step (i): Converting the trigonometrical values of the longitude and latitude into radian.
# 
# 
# Step (ii): Find the difference in the coordinates.
# 
# 
# Step (iii): Use one of the formulars above to calculate the distance between two points.
# 

# In[34]:


import math
from math import sin, cos, sqrt, atan2, radians
R = 6373.0 # Mean Radius of the Earth

# Step(i) Converting the trigonometrical values of the longitude and latitude into radian.
lat_x_rad = math.radians(lat_x)
lon_x_rad= math.radians(lon_x)
lat_y_rad = math.radians(lat_y)
lon_y_rad= math.radians(lon_y)

# Step(ii) Find the difference in the coordinates.
diff_lon = lon_y_rad - lon_x_rad
diff_lat = lat_y_rad - lat_x_rad

# Step(iii)  For the purpose of this assignment,the Haversine formula would be used. 
# Using Haversine formula to calculate the distance between two points.
a = math.sin(diff_lat / 2)**2 + math.cos(lat_x_rad) * math.cos(lat_y_rad) * math.sin(diff_lon / 2)**2
c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
dist = R * c

print("The distance (in meters) between the cheapest sale and the most recent sale:", dist* 1000, 'metres')


# .

# (B) what is the median street number, in multi-family houses, sold between 05/11/1933 and 03/12/1998 , in Sacramento?
# 
# 
# SOLUTION / APPROACH
# 
# 
# To do this:
# 
# 
# STEP 1: Filter out the rows with 'city= SACRAMENTO' and 'type= MULTI-FAMILY' 

# In[35]:


# Filter out the rows with 'city= SACRAMENTO' and 'type= MULTI-FAMILY' , 
data_add=new_data[(new_data['type']=='MULTI-FAMILY') & (new_data['city']=='SACRAMENTO')]
data_add


# 
# 
# 
# STEP 2: Filter the date that falls between '05/11/1933' and '03/12/1998' in step 1

# In[36]:


# From the data_add gotten above, fiter the date thaat falls between 05/11/1933 and 03/12/1998
date_filter = (data_add['sale_date'] > '1933-11-05 00:00:00') & (data_add['sale_date'] <= '1998-12-03 00:00:00') # Filter date 05/11/1933 and 03/12/1998
data_ctd= data_add.loc[date_filter] # data with filtered city= SACRAMENTO, type=MULTI-FAMILY and date=05/11/1933 and 03/12/1998.
data_ctd


#  STEP 3: From the 'street' column, extract the characters before the first blankspace. This corresponds to the street numbers .Then, find the median of these numbers

# In[37]:


# Extract street numbers from the street column (by splitting the content of the column by blank spaces and extracting the first value) 
# The result is passed to the median value method 
street_num = (data_ctd['street'].apply(lambda x: x.split()[0])).median()
print('The median street number, in multi-family houses, sold between 05/11/1933 and 03/12/1998 , in Sacramento is: ',street_num)


# .

# (C) What is the city name, and its 3 most common zip codes, that has the 2nd highest amount of beds sold?

# SOLUTION / APPROACH
# 
# 
# To do this:
# 
# 
# STEP 1: Get the name of the city that has the 2nd highest amount of beds sold
# This is achieved by summing the number of beds per city. 
# 
# The name of the city with the second highest number of sold beds is gotten

# In[38]:


# Step 1: Get the name of the city that has the 2nd highest amount of beds sold
# This is achieved by 
k=new_data.groupby('city')['beds'].sum()
k.nlargest(2).iloc[[-1]]


# STEP 2: Find the three (3) most common zip codes of the city(ELK GROVE) gotten in step 1 
# 

# In[39]:


# Filter out ELK GROVE rows from th original data since we established that ELK GROVE is the city of interest.
data_elk=new_data[(new_data['city']=='ELK GROVE')]
data_elk


#  STEP 3 Do a groupby of zip with the GROVE city.This gives all the unique zip codes belonging to ELK GROVE
# 
# 
# Then, count the number of occurrences(frequency) of the unique ELK GROVE's zip codes and rename the resulting column as frequency
# 
# Rearrange the table in descending order 

# In[40]:


data_elk.groupby(['city','zip']).size().reset_index(name='frequency').sort_values(['frequency','zip'],ascending=[0,1]).groupby('city').head(3)


# In[41]:


stg='''Therefore, the city name, and the 3 most common zip codes, that has the 2nd highest amount of beds sold: \n 
city name: ELK GROVE \n
Zip codes: 95758,95757 and 95624'''
print(stg)

