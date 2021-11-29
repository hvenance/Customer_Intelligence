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






#k-means algorithm
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(new_df.iloc[:,[1,2]].values)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(new_df.iloc[:,[1,2]].values)

#count the number of customers in each cluster
for i in range(4):
    print(np.sum(y_kmeans == i))

# Visualising the clusters
plt.scatter(new_df.iloc[:,[1,2]].values[y_kmeans == 0, 0], new_df.iloc[:,[1,2]].values[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(new_df.iloc[:,[1,2]].values[y_kmeans == 1, 0], new_df.iloc[:,[1,2]].values[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(new_df.iloc[:,[1,2]].values[y_kmeans == 2, 0], new_df.iloc[:,[1,2]].values[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(new_df.iloc[:,[1,2]].values[y_kmeans == 3, 0], new_df.iloc[:,[1,2]].values[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(new_df.iloc[:,[1,2]].values[y_kmeans == 4, 0], new_df.iloc[:,[1,2]].values[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 175, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers via K-means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


import scipy.cluster.hierarchy as sch
method = ['ward', 'single', 'complete', 'average']

for i in method:
    dendrogram = sch.dendrogram(sch.linkage(new_df.iloc[:,1:3], method = i))
    # Ward's method: distance between two clusters A and B is how much the sum of squares will increase when we merge them
    # single for min distance
    # complete for max distance
    # average for centroid distance
    plt.title('Dendrogram using method '+ i)
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()


dendrogram = sch.dendrogram(sch.linkage(new_df.iloc[:,1:3], method = 'ward'))
        # Ward's method: distance between two clusters A and B is how much the sum of squares will increase when we merge them
        # single for min distance
        # complete for max distance
        # average for centroid distance
        plt.title('Dendrogram using method ')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(new_df.iloc[:,1:3])

X=new_df.iloc[:,[1,2]].values
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


#if __name__ == '__main__':
    # Override default pandas configuration
    #pd.options.display.width = 0
    #pd.options.display.max_rows = 10000
    #pd.options.display.max_info_columns = 10000
    #df = data
    #prof = pandas_profiling.ProfileReport(df=df)
    #prof.to_file('pandas_profile_test.html')


