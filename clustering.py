from sklearn.cluster import KMeans
# from data_analysis import new_df
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

new_df = pd.read_csv('C:/Users/Hadrien Venance/customerIntelligence/MarketSegmentation/spending_frequency_score.csv', index_col=[0])

#Feature scaling


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(np.array(new_df.iloc[:, 1:3]))
    wcss.append(kmeans.inertia_)

#elbow method
fig = go.Figure()
fig = px.line(
    x=range(1,11),
    y=wcss,
)
fig.update_layout(
    title="The Elbow Method",
    xaxis_title="Number of clusters",
    yaxis_title="WCSS",
)
fig.show(renderer="browser")


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(np.array(new_df.iloc[:, 1:3]))	
y_kmeans = [str(prediction) for prediction in y_kmeans]

fig = px.scatter(
    data_frame= new_df,
    x='spending_score', y='frequency_score', color=y_kmeans)

fig.show(renderer='browser')

#count instances in each cluster
for i in range(4):
    print(np.sum(y_kmeans == i))


# kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]


# fig = go.Figure()

# fig.add_traces(go.Scatter(
# 				x=new_df['spending_score'],
# 				y=new_df['frequency_score'],
# 				mode='markers',
# 				marker=dict(color=y_kmeans, size=10),
# 				name='Clusters'
# 	))

# fig.add_traces(go.Scatter(
# 				x=kmeans.cluster_centers_[:, 0],
# 				y=kmeans.cluster_centers_[:, 1],
# 				mode='markers',
# 				marker=dict(color='green', size=30),
# 				name='Clusters centroids'
# 	))

# fig.show()



#hierarchical clustering
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



import plotly.figure_factory as ff
# fig = ff.create_dendrogram(new_df.iloc[:, 1:3],
#                            linkagefun=lambda x: sch.linkage(x, "single"), )
# fig.update_layout(width=800, height=500,
#                   title=('Dendrogram using method single'),
#                   xaxis_title="Customers",
#                   yaxis_title="Height",
#                   )
# fig.show(renderer="browser")

for i in method:
    fig = ff.create_dendrogram(new_df.iloc[:,1:3],
                               linkagefun=lambda x: sch.linkage(x, i),)
    fig.update_layout(width=800, height=500,
        title = ('Dendrogram using method '+ "i"),
        xaxis_title = "Customers",
        yaxis_title = "Height",
    )
    fig.show(renderer="browser")




# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(new_df.iloc[:,1:3])
y_hc = [str(prediction) for prediction in y_hc]

#plotly visualisation
fig = px.scatter(
    data_frame= new_df,
    x='spending_score', y='frequency_score', color=y_hc)
fig.show(renderer='browser')

import plotly.graph_objects as go
fig = go.Figure()
fig.add_traces(go.Scatter(
				x=new_df['spending_score'],
				y=new_df['frequency_score'],
				mode='markers',
				marker=dict(color=0, size=10),
				name='Frugal'
	))
fig.add_traces(go.Scatter(
				x=new_df['spending_score'],
				y=new_df['frequency_score'],
				mode='markers',
				marker=dict(color=y_hc , size=10),
				name='Regular'
	))
fig.add_traces(go.Scatter(
				x=new_df['spending_score'],
				y=new_df['frequency_score'],
				mode='markers',
				marker=dict(color=y_hc, size=10),
				name='Loyal'
	))
fig.add_traces(go.Scatter(
				x=new_df['spending_score'],
				y=new_df['frequency_score'],
				mode='markers',
				marker=dict(color=3, size=10),
				name='Ambassador'
	))
fig.update_layout(width=800, height=500,
        title = ('Cluster of customers'),
        xaxis_title = "spending_score",
        yaxis_title = "frequency_score",
    )
fig.show(renderer="browser")


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



# new_df = pd.read_csv('C:/Users/Hadrien Venance/customerIntelligence/MarketSegmentation/spending_frequency_score.csv')
#
# from sklearn.cluster import AffinityPropagation
# np.median(new_df.iloc[:,1:3])
# clustering = AffinityPropagation(random_state=5, damping=0.5, max_iter=2500).fit(new_df.iloc[:,1:3])
# clustering.labels_
# cluster_centers_indices = clustering.cluster_centers_indices_
# len(cluster_centers_indices)
# affinity = clustering.predict(new_df.iloc[:,1:3])
#
# #graph of affinity propagation
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', affinity))
# fig = px.scatter(
#     data_frame= new_df,
#     x='spending_score', y='frequency_score', color=colors)
#
# fig.show()
#
