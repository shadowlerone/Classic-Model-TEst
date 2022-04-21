import model
import test

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


sites = ['FI-Hyy', 'GH-Ank', 'CA-TVC']

temps = [0 + i/2 for i in range(20)]
moistures = [0 + i/2 for i in range(10) ]

df = pd.read_csv('./data.csv')


figure, axis = plt.subplots(3,1)
i = 0 
figure.set(size_inches = (8,8))

# print(df[df.site == 'FI-Hyy'].temperature)

for site in sites:
	site_df = df[df.site == site]
	# axis[i,0].plot(site_df.temperature, site_df.respiration)
	site_df.plot(ax=axis[i], x='temperature', y='respiration', kind='scatter')
	axis[i].set_title(site)
	i += 1

plt.savefig('test_graphs/multiplots.png')
plt.rc('pgf', texsystem='xelatex')
plt.savefig('test_graphs/multiplots.pgf')