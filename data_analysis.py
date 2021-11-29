#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
#import pandas_profiling
import matplotlib
matplotlib.use('qt5agg')

#load dataset
data = pd.read_csv("C:/Users/Hadrien Venance/customerIntelligence/MarketSegmentation/or.csv")
print(data.head())

#clean the dataset
print(" \nshow the boolean Dataframe : \n\n", data.isnull())

# Count total NaN at each column in a DataFrame
print(" \nCount total NaN at each column in a DataFrame : \n\n",
      data.isnull().sum())

#drop the clients without customer_id
data = data.dropna(subset=['cid'])

max(data['price'])
min(data['price'])
abs(max(data['price'])-min(data['price']))

#we see that no prices are negative anymore
np.sum(np.array(data['price'])<= 0)


#creating spending_score variable
spending_df = data.iloc[:, [5,6]]
spending_df.head()

spending_score = spending_df.groupby('cid').sum()
print(spending_score)
spending_score = spending_score.reset_index()
print(spending_score)
spending_score.iloc[:,0]
spending_score.iloc[:,1]
missing_spending = spending_score[spending_score['cid'].isnull()]

#according to our boxplot (see graph_data_analysis file) the upper bound of the spending score should be 1061.65, the rest of the observations is outliers.
spending_score = spending_score[spending_score['price'] <= 1061.65]


#creating frequency_score variable
frequency_df = data.iloc[:, [1,6]]
frequency_df.head()

frequency_score = frequency_df.groupby('cid').count()
print(frequency_score)
type(frequency_score)

#according to our boxplot the upper bound of the frequency score should be 328, the rest of the observations is outliers.
frequency_score = frequency_score[frequency_score['inv'] <= 328]


#new dataframe to combine the 2 newly created variables
new_df = pd.merge(frequency_score, spending_score, on='cid')
new_df.head()

#rename columns of this new_df
new_df = new_df.rename(columns={'cid':'customer_ID', 'inv':'frequency_score', 'price':'spending_score'})
new_df.head()

#to write the new database containing frequency and spending score on a new csv file
new_df.to_csv('spending_frequency_score.csv')




#if __name__ == '__main__':
    # Override default pandas configuration
    #pd.options.display.width = 0
    #pd.options.display.max_rows = 10000
    #pd.options.display.max_info_columns = 10000
    #df = data
    #prof = pandas_profiling.ProfileReport(df=df)
    #prof.to_file('pandas_profile_test.html')


