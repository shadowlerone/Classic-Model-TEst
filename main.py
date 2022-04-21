import model
import test

from concurrent.futures.thread import ThreadPoolExecutor
import time

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm

import argparse

from os import path, mkdir

import warnings
warnings.filterwarnings("ignore") ### WARNING. THIS LINE OF CODE IS NOT RECOMMENDED. I'M USING IT HERE BECAUSE I'M SICK OF THESE ERRORS PRINTING FOR EVERY LOOP.

parser = argparse.ArgumentParser(
	description='Run and analyse CLASSIC Climate model.'
)

parser.add_argument("-r", "--run", help="run the climate model", action="store_true")
parser.add_argument("-a", "--analyse", help="runs climate model analysis", action="store_true")

parser.add_argument("-s", "--sites", type=str, nargs='+', choices=['FI-Hyy', 'GH-Ank', 'CA-TVC'])

parser.add_argument("-i", "--iters", type=int, default=80000*2)

args = parser.parse_args()

sites = args.sites or ['FI-Hyy', 'GH-Ank', 'CA-TVC']

temps = [-10 + i/2 for i in range(40)]
moistures = [0 + i/2 for i in range(10) ]

retests = 1



# zeroDegreeTest = test.Test(
# 	'zero degree test', 
# 	'./', 
# 	'FI-Hyy', 
# 	'Zero Degree Test - TEST', 0, 0
# )
data = []

if args.run:

	for site in sites:
		x = []
		y = []
		# if site == 'CA-TVC':
		# 	i = args.iters * 2
		# else:
		# 	i = args.iters
		for temperature in temps:
		# temperature = 0
			for r in range(retests):
				x.append(temperature)
				y.append(
					round(
						np.mean(
							np.sum(
								model.Model.test(
									test.Test(
										'zero degree test', 
										'./', 
										site, 
										f"{site} - {temperature} - {0}", temperature, 
										0,
										maxiters=args.iters
									)
								), 
								axis=(0,1,2)
							)
						), 3
					)
				)
		data.append(
			{
				'site': site,
				'temperature': x,
				'respiration': y
			}
		)
		# print(x)
		# print(round(np.mean(np.sum(y, axis=(0,1,2))), 3))
		# print(y)
		# plt.plot(x,y , label=f'Respiration - {site}')
if args.analyse or args.run:
	plt.cla()

	# print(data)
	# d2 = [[i['site'], i['x'], i['y']] for i in data]
	# print(d2)
	if args.run:
		for i in data:
			plt.scatter(i['temperature'],i['respiration'] , label=f'Respiration - {i["site"]}')
		df = pd.DataFrame.from_dict(data).explode(['temperature','respiration'])
		df.to_csv('./data.csv')
		plt.title("Carbon Respiration by Temperature Increase")
	
		plt.xlabel("Temperature increase")
		plt.ylabel("Average Respiration")

		plt.legend()


		plt.savefig('test_graphs/overall.png')
		plt.rc('pgf', texsystem='xelatex')
		plt.savefig('test_graphs/overall.pgf')

		plt.cla()
	else: 
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
	for site in sites:
		site_df = df[df.site == site]
		model = ols("respiration ~ temperature", site_df).fit()
		print(model.summary())
		pp = path.join('test_data', 'summaries')
		p = path.join('test_data', 'summaries', f"{site}.tex")
		if (not path.isdir(pp)):
				mkdir(pp)
		with open(p, 'w') as f:
			[f.write(i.as_latex_tabular()) for i in model.summary().tables]

# model.Model.test(zeroDegreeTest)