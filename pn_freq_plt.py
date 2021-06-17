import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys


import matplotlib
from matplotlib.dates import DateFormatter, MonthLocator

import os

# os.chdir(os.getcwd())

# workdir = r'C:\Users\YU Yang\Desktop\PUBG_freq'
# os.chdir(workdir)


df_p_freq = pd.read_csv('./outputs/topics_pos_freq.csv',engine='c',index_col=0)
df_n_freq = pd.read_csv('./outputs/topics_neg_freq.csv',engine='c',index_col=0)


df_p = pd.read_csv('./outputs/topics_pos.csv',engine='c',index_col=0)
df_n = pd.read_csv('./outputs/topics_neg.csv',engine='c',index_col=0)

assert df_p_freq.columns.values.all() == df_p_freq.columns.values.all(), "(Frequency)Topic words are different!"
assert df_p_freq.index.values.all() == df_p_freq.index.values.all(), "(Frequency)Dates are different!"

assert df_p.columns.values.all() == df_p.columns.values.all(), "(Value)Topic words are different!"
assert df_p.index.values.all() == df_p.index.values.all(), "(Value)Dates are different!"

for topic_word in df_p_freq.columns:
    word_pos = df_p[topic_word]
    word_neg = df_n[topic_word]

    x_axis = df_p.index.values

    ticker_spacing = x_axis
    ticker_spacing = 12

    y_pos = word_pos.values
    y_neg = word_neg.values


    pos_rate_lst = []

    for pos, neg in zip(y_pos, y_neg):
        
        if pos == 0 and neg == 0:
            pos_rate = 0
        else:
            pos_rate = (pos / (pos + neg)) * 100

        
        pos_rate_lst.append(pos_rate)

    fig, ax1 = plt.subplots()


    # ax1.set_xlabel('date (month) ')
    # ax1.set_ylabel('frequency')


    ax1.plot(df_p_freq[topic_word], color='tab:blue')
    ax1.plot(df_n_freq[topic_word], color='tab:red')
    # ax1.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


    # ax2.set_ylabel('positive (%)')  # we already handled the x-label with ax1
    ax2.plot(x_axis, pos_rate_lst, color='tab:green', linestyle="--")
    ax2.tick_params(axis='y')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())


    ax2.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))

    plt.title('Topic: {}'.format(topic_word))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig('figures/{}.png'.format(topic_word.split('/')[0]))

plt.show()
