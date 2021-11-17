import pandas as pd
import plotly.express as px
from plotly.offline import plot
import numpy as np 
import plotly.figure_factory as ff
import os


# IMPORT DB 
df = pd.read_csv('or.csv') 

new_df = pd.read_csv('spending_frequency_score.csv')



# EXPLORATION
print(df.describe())



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

    # figure.write_image(f"images/{output_name}.png")
    figure.show()

# export_hist(df=df, feature='price', title="Histogram of the orders' price.", nbins=40, output_name='hist_price_raw_data')

# export_hist(df=df[df['price'] > 0], feature='price', log_y=True,
#                                             title="Logarithmic histogram of the orders' price withtout the outliers.",
                                            # nbins=40,
                                            # output_name='hist_price_without_outliers')

# export_hist(df=df[df['q'] != 0], feature='q', title="Histogram of the orders' quantity bought.", log_y=True, nbins=40, output_name='hist_quantity')

# export_hist(df=new_df, feature='spending_score', title='Histogram of spending score', nbins=40, log_y=False, output_name="hist_spending_score")

# export_hist(df=new_df, feature='frequency_score', title='Histogram of frequency score', nbins=40, log_y=False, output_name="hist_freq_score")


# if __name__ == "__main__":
#     fig = ff.create_distplot([df['price']], ['Price'])
#     fig.write_image(f"images/dist_plot_basic.png")