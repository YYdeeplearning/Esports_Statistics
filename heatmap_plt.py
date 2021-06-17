import pandas as pd
import datetime as dt
import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter, MonthLocator

df_freq = pd.read_csv('./outputs/topics_frequency3.csv',engine='c',index_col=0)

plt.figure(figsize = (20,7.5))

sns.heatmap(df_freq, cmap="RdBu_r")
#sns.heatmap(df_freq, cmap = "Spectral_r")
#sns.heatmap(df_freq, cmap = "RdYlBu_r")
#sns.heatmap(df_freq, cmap = "seismic")

#x_axis = df_freq.columns.values

#ticker_spacing = x_axis
#ticker_spacing = 12

#fig, ax1 = plt.subplots()


#ax1.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))

plt.title('Topics_heatmap')
plt.xlabel('Month')
plt.ylabel('Topics')

plt.savefig('./heatmaps/heatmap.png')
#plt.savefig('./heatmaps/heatmap.pdf')
plt.show()
