import pandas as pd

import matplotlib.ticker as ticker
import sys

import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import joblib
from gensim.corpora import MmCorpus, Dictionary
from wordcloud import WordCloud
import sys
import os
import numpy as np

import matplotlib
from matplotlib.dates import DateFormatter, MonthLocator


workdir = r'C:\Users\YU Yang\Desktop'
os.chdir(workdir)



df_p_rate = pd.read_csv('topics_pos_rate100.csv',engine='c',index_col=0)

df_p_rate


df1 = df_p_rate[['in-game purchases', 'region','community','optimization','matchmaking']]
df2 = df_p_rate[['cheating','server','maps','gameplay','character design']]
df3 = df_p_rate[['graphics','learning curve','player skills','teamwork']]



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,15))

plt.grid(True)


df1.plot(ax=axes[0], style = ["-","--","-.",".-",":"])
df2.plot(ax=axes[1], style = ["-","--","-.",".-",":"])
df3.plot(ax=axes[2], style = ["-","--","-.",".-"])

axes[0].tick_params(axis='y')
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())

axes[0].grid()

axes[1].tick_params(axis='y')
axes[1].yaxis.set_major_formatter(ticker.PercentFormatter())

axes[1].grid()

axes[2].tick_params(axis='y')
axes[2].yaxis.set_major_formatter(ticker.PercentFormatter())

axes[2].grid()

axes[0].set_title('Topics_trend')
axes[1].set_ylabel('Positive rate(%)')
axes[2].set_xlabel('Month')

# ax1.set_xlabel('date (month) ')
# ax1.set_ylabel('frequency')

plt.savefig('topics_trend.png')
# plt.savefig('topics_trend.pdf')

plt.show()