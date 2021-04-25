#!/usr/bin/env python
# coding: utf-8

# UCD Certificate in Introductory Data Analytics

# 
# USA Name Data | Kaggle https://www.kaggle.com/datagov/usa-names
# https://raw.githubusercontent.com/organisciak/names/master/data/us-names-by-decade.csv



import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from numpy import median, mean



us_names_df = pd.read_csv('https://raw.githubusercontent.com/organisciak/names/master/data/us-names-by-decade.csv')
us_names_df.head()


# Using inbuilt pandas functions to analyze our data - .info and .head. Using shape to count the number of observations.
# I have checked for null values



us_names_df.info()
us_names_df.shape
missing_values_count = us_names_df.isnull().sum()

print(missing_values_count[0:4])


# For our graphs we will use the Seaborne library.First we will  find out and show which decade had the most names.First  we need to work our how many decades we have. our column decade shows 11 decades.This produces an array - a list of data of the same type.Using a for loop and the loc function to iterate over our dataframe I have created a dictionary to show us the number of names in each decade. 


us_names_df["decade"].unique()





def decadedict():
    decade = [1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
    name_count={}
    for x in decade:
        value =us_names_df.loc[us_names_df["decade"]==x].decade.value_counts()
        name_count[x] = value[x]
    return name_count




print(decadedict())




g = sns.catplot('decade', data=us_names_df, kind="count", aspect=3);
plt.title("Which decade had the MOST names");
sns.set_context("poster")


# To combine my data frames I select concat . I have created two data frames - the top 5 names in each decade



#top_us_names_df['index'] =top_us_names_df.reset_index(level=0, inplace=True)
top_us_names_df=us_names_df.groupby('decade').head(5)
top_us_names_df.head(55)





bottom_us_names_df=us_names_df.groupby('decade').tail(5)
#bottom_us_names_df.tail(55)





concat_data= pd.concat([top_us_names_df,bottom_us_names_df])






print(concat_data.shape)


#  The average number of names per decade.




sns.set(style="whitegrid")
plt.figure(figsize = (12, 8))
plt.title("The average number of names per decade")
sns.barplot(x="decade", y="count", data=us_names_df, estimator=mean);
sns.set_style("whitegrid")
sns.set_context("poster")
("")


#  US graduate schools admissions data is from Kaggle
# #https://www.kaggle.com/tanmoyie/us-graduate-schools-admission-parameters/download




df = pd.read_csv("US_graduate_schools_admission_parameters_dataset.csv")





df.info()





#df =df.head(40)


# on 40 the correlation is .8526 but on full dataset it is .66 ran graphs on full data set




df.describe()





x = np.array(df['GRE Score'])
y = np.array(df['University Rating'])
r = np.corrcoef(x, y)
print(r)
corr=r





sns.set_style("darkgrid")
sns.set_context("poster")
#sns.set_palette("rainbow")
plt.figure(figsize=(11,8))
sns.scatterplot(data=df,x="University Rating", y = "GRE Score", alpha =0.9, size = "CGPA" ,hue="CGPA");
plt.legend(loc="upper right",bbox_to_anchor=(1.4,1));
plt.title("Graduate Admission to top Universities based on Academic Scores")


# Data set number 3 is called Penguis from the Seaborne library.I  selected this as neither of the above libraries has any null values. I renamed column sex to gender
# 
# https://github.com/mwaskom/seaborn-data
# 




df1 = sns.load_dataset("penguins").rename(columns={"sex":"gender"})





df1





missing_values_count = df1.isnull().sum()
print(missing_values_count[0:8])





# drop rows where data is missing. 11 rows with no data
droprows = df1.dropna()
print(df1.shape,droprows.shape)





penguins=droprows





penguins.info()





# resize plot and set style
sns.set_style("darkgrid")
sns.set_context("poster")
#plt.title("The average number of names per decade")
plt.figure(figsize = (11,8))
sns.scatterplot(data=penguins, x = "body_mass_g" , y = "bill_length_mm",alpha=0.9,hue = "species",size = "gender");
plt.legend(loc = "upper right",bbox_to_anchor=(1.4,1));
plt.title("Penguin Species")





