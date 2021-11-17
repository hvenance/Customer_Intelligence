from sklearn.cluster import KMeans
from data_analysis import new_df
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(np.array(new_df.iloc[:, 1:3]))
#     wcss.append(kmeans.inertia_)


# fig = px.line(x=range(1,11), y=wcss)
# fig.show()

# scaler = MinMaxScaler()


# new_df.iloc[:, 1:3] = scaler.fit_transform(new_df.iloc[:, 1:3])

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(np.array(new_df.iloc[:, 1:3]))	
y_kmeans = [str(prediction) for prediction in y_kmeans]

fig = px.scatter(data_frame= new_df, x='spending_score', y='frequency_score', color=y_kmeans)
fig.show()



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