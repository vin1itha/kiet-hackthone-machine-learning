import numpy as np         # linear algebra
import pandas as pd        # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the file "data.csv" and print the contents.
#!cat data/data.csv
data = pd.read_csv('data/data.csv', index_col=False,)


# #### Load Dataset
# 
# First, load the supplied CSV file using additional options in the Pandas read_csv function. 

# #### Inspecting the data
# The first step is to visually inspect the new data set. There are multiple ways to achieve this:
# * The easiest being to request the first few records using the DataFrame data.head()* method. By default, “data.head()” returns the first 5 rows from the DataFrame object df (excluding the header row). 
# * Alternatively, one can also use “df.tail()” to return the five rows of the data frame. 
# * For both head and  tail methods, there is an option to specify the number of records by including the required number in between the parentheses when calling either method.Inspecting the data

# In[2]:


data.head(2)


# You can check the number of cases, as well as the number of fields, using the shape method, as shown below.

# In[3]:


# Id column is redundant and not useful, we want to drop it
data.drop('id', axis =1, inplace=True)
#data.drop('Unnamed: 0', axis=1, inplace=True)
data.head(2)


# In[4]:


data.shape


# In the result displayed, you can see the data has 569 records, each with 32 columns.
# 
# The **“info()”** method provides a concise summary of the data; from the output, it provides the type of data in each column, the number of non-null values in each column, and how much memory the data frame is using.
# 
# The method **get_dtype_counts()** will return the number of columns of each type in a DataFrame:

# In[5]:


# Review data types with "info()".
data.info()


# In[6]:


# Review number of columns of each data type in a DataFrame:
data.get_dtype_counts()


# From the above results, from the 32, variables,column id number 1 is an integer, diagnosis 569 non-null object. and rest are float. More on [python variables](https://www.tutorialspoint.com/python/python_variable_types.htm)

# In[7]:


#check for missing variables
#data.isnull().any()


# In[8]:


data.diagnosis.unique()


# From the results above, diagnosis is a categorical variable, because it represents a fix number of possible values (i.e, Malignant, of Benign. The machine learning algorithms wants numbers, and not strings, as their inputs so we need some method of coding to convert them.
# 
# 

# In[9]:


#save the cleaner version of dataframe for future analyis
data.to_csv('data/clean-data.csv')


# > ### Now that we have a good intuitive sense of the data, Next step involves taking a closer look at attributes and data values. In nootebook title :NB_NB2_ExploratoryDataAnalys, we will explore the data further
