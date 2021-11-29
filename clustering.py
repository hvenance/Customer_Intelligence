from sklearn.cluster import KMeans
from sklearn.preprocessing import  MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import os

# IMPORT DB 
df = pd.read_csv('or.csv') 

new_df = pd.read_csv('spending_frequency_score.csv')


# EXPLORATION
# print(df.describe())



if not os.path.exists("images"):
    os.mkdir("images")

def export_hist(df, feature, title, output_name, log_y=False, nbins=40, box_plot=True):
    figure = px.histogram(
                        df,
                        x=feature,
                        title=title,
                        log_y=log_y,
                        nbins=nbins,
                        marginal= 'box' if box_plot else None
        )

    figure.write_image(f"images/{output_name}.png")
    # figure.show()

export_hist(df=df, feature='price', title="Histogram of the orders' price.", nbins=40, output_name='hist_price_raw_data')

export_hist(df=df[df['price'] > 0], feature='price', log_y=True,
                                            title="Logarithmic histogram of the orders' price withtout the outliers.",
                                            nbins=40,
                                            output_name='hist_price_without_outliers')

export_hist(df=df[df['q'] != 0], feature='q', title="Histogram of the orders' quantity bought.", log_y=True, nbins=40, output_name='hist_quantity')

export_hist(df=new_df, feature='spending_score', title='Histogram of spending score', nbins=40, log_y=False, output_name="hist_spending_score")

export_hist(df=new_df, feature='frequency_score', title='Histogram of frequency score', nbins=40, log_y=False, output_name="hist_freq_score")




wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(np.array(new_df.iloc[:, 1:3]))
    wcss.append(kmeans.inertia_)


fig = px.line(x=range(1,11), y=wcss)
fig.show()

scaler = MinMaxScaler()


new_df.iloc[:, 2:4] = scaler.fit_transform(new_df.iloc[:, 2:4])

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(np.array(new_df.iloc[:, 2:4]))	
y_kmeans = np.array([str(prediction) for prediction in y_kmeans])

# fig = px.scatter(data_frame= new_df, x='spending_score', y='frequency_score', color=y_kmeans)
# fig.show()



# kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]


spending_score = new_df['spending_score']
frequency_score = new_df['frequency_score']
cluster_1 = new_df[y_kmeans == '0']
cluster_2 = new_df[y_kmeans == '1']
cluster_3 = new_df[y_kmeans == '2']
cluster_4 = new_df[y_kmeans == '3']

print('frugal', len(cluster_3))
print('regular', len(cluster_1))
print('loyal', len(cluster_4))
print('ambassador', len(cluster_2))




fig = go.Figure()


fig.add_traces(go.Scatter(
				x=cluster_3['spending_score'],
				y=cluster_3['frequency_score'],
				mode='markers',
				opacity=0.85,
				marker=dict(color=3, size=5),
				name='Frugals' 
	))


fig.add_traces(go.Scatter(
				x=cluster_1['spending_score'],
				y=cluster_1['frequency_score'],
				mode='markers',
				marker=dict(color=1, size=5),
				name='Regular' 	
	))

fig.add_traces(go.Scatter(
				x=cluster_4['spending_score'],
				y=cluster_4['frequency_score'],
				mode='markers',
				marker=dict(color=4, size=5),
				name='Loyals'
	))

fig.add_traces(go.Scatter(
				x=cluster_2['spending_score'],
				y=cluster_2['frequency_score'],
				mode='markers',
				marker=dict(color=2, size=5),
				name='Ambassadors'
	))

fig.add_traces(go.Scatter(
				x=kmeans.cluster_centers_[:, 1],
				y=kmeans.cluster_centers_[:, 0],
				mode='markers',
				marker=dict(color=5, size=15),
				name='Clusters centroids'
	))

fig.update_layout(
			title="Clusters Visualisation with KMeans.",
    		xaxis_title="Spending Score.",
		    yaxis_title="Frequency Score",
		    legend_title="Legend:"
	)

fig.show()
fig.write_image(f"images/predictions.png")


hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'average')
y_hc = hc.fit_predict(new_df.iloc[:,2:4])
y_hc = np.array([str(prediction) for prediction in y_hc])



cluster_1 = new_df[y_hc == '0']
cluster_2 = new_df[y_hc == '1']
cluster_3 = new_df[y_hc == '2']
cluster_4 = new_df[y_hc == '3']



fig = go.Figure()
fig.add_traces(go.Scatter(
				x=cluster_1['spending_score'],
				y=cluster_1['frequency_score'],
				mode='markers',
				marker=dict(color=0, size=10),
	))
fig.add_traces(go.Scatter(
				x=cluster_2['spending_score'],
				y=cluster_2['frequency_score'],
				mode='markers',
				marker=dict(color=1, size=10),
	))
fig.add_traces(go.Scatter(
				x=cluster_3['spending_score'],
				y=cluster_3['frequency_score'],
				mode='markers',
				marker=dict(color=2, size=10),
	))
fig.add_traces(go.Scatter(
				x=cluster_4['spending_score'],
				y=cluster_4['frequency_score'],
				mode='markers',
				marker=dict(color=3, size=10),
	))
fig.update_layout(
        title = ('Cluster of customers'),
        xaxis_title = "spending_score",
        yaxis_title = "frequency_score",
    )
fig.show(renderer="browser")
