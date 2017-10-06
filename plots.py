import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import helpers
import os
import pandas as pd
import scipy
import fwd_sample as fs
import fwd_likelihoods as fl
import scipy

def timeseries_predict_plot(data, thetas, predict_index, num_samples=100, tmax=60*60):
	k1 = 0.00055
	theta7 = np.power(10., thetas[6])
	theta8 = np.power(10., thetas[7])
	fig, axes = plt.subplots(2,1, figsize=(10,10))

	bout_ts_holder = [[0]]
	stomach_ts_holder = [[0]]

	## Generate the base time series
	for event in data:
		f_length, g_start, rate, p_length, g_end_feeding = event

		## Bout ts holder gets one entry at rate for time f_length
		## and one entry at rate 0 for time p_length
		bout_ts_holder.append(rate*np.ones(int(f_length)))
		bout_ts_holder.append(np.zeros(int(p_length)))

		## Stomach ts holder gets a linearly increasing line at rate for f_length
		## then decreases according to the ODE
		g_start = stomach_ts_holder[-1][-1]
		stomach_ts_holder.append(g_start + 3.5*rate*(1+np.arange(int(f_length)))) # feeding

		g_end = stomach_ts_holder[-1][-1]
		t_c = 2.*np.sqrt(g_end)/k1		
		
		if p_length <= t_c:
			digestion_ts = 0.25*np.power((2.*np.sqrt(g_end) - k1*(1+np.arange(int(p_length)))), 2)
			stomach_ts_holder.append(digestion_ts)

		else:
			digestion_ts1 = 0.25*np.power((2.*np.sqrt(g_end) - k1*(1+np.arange(int(t_c)))), 2)
			digestion_ts2 = np.zeros(int(p_length - t_c))
			stomach_ts_holder.append(digestion_ts1)
			stomach_ts_holder.append(digestion_ts2)

	## Create a time series from the time series holder
	bout_ts = np.hstack(bout_ts_holder)
	stomach_ts = np.hstack(stomach_ts_holder)

	## Plot the time series
	axes[0].plot(bout_ts)
	axes[1].plot(stomach_ts)

	## Now do the posterior prediction of next mealtime
	## Get the time of the start of the intermeal interval we wish to predict
	known_events = data[:predict_index]
	start_time = 0
	for event in known_events[:-1]: # only want feeding from the last one
		start_time += event[0] + event[3]

	start_time += known_events[-1][0] # add the g_end_feeding
	x0 = stomach_ts[int(start_time)]

	next_times = []
	i = 0
	while i < num_samples:
		new_sample = fl.sample_L(x0, k1, theta7, theta8)
		next_times.append(start_time + new_sample)

		i += 1

	axes[0].set_ylim(0, 0.02)
	ax2 = axes[0].twinx()

	## Histogram of results
	#ax2.hist(next_times, histtype='step', color='r', bins=100, normed=True)

	## KDE of results
	x_grid = np.arange(len(stomach_ts))
	kde = scipy.stats.gaussian_kde(next_times)
	y = kde.evaluate(x_grid)
	ax2.plot(x_grid, y, c='r')
	ax2.fill_between(x_grid, 0, y, color='r', alpha=0.3)
	ax2.set_ylim([0, 2.5*np.max(y)])

	## Predict samples of stomach fullness
	ts_predictions = []
	i = 0
	while i < num_samples:
		new_sample = fs.sample(tmax, thetas, x0, init_state='L')
		ts_predictions.append(new_sample[0][:tmax])
		i += 1

	ts_predictions = np.stack(ts_predictions)
	x = start_time + np.arange(tmax) # offset to start time

	## Plot mean
	mean_ts = np.mean(ts_predictions, axis=0)
	axes[1].plot(x, mean_ts, c='r')

	## Plot percentile
	pc = 5
	min_val = np.percentile(ts_predictions, pc, axis=0)
	max_val = np.percentile(ts_predictions, 100-pc, axis=0)
	axes[1].fill_between(x, min_val, max_val, alpha=0.3, color='r')

	return fig, axes



		


def pairplot(df, vars, c, ms, fig, axes):
	return 0


def trellisplot(df, varlist, fig, axes):
	## Prepare the figure
	num_vars = len(varlist)

	## Plot the data
	for i, var1 in enumerate(varlist):
		for j, var2 in enumerate(varlist):

			## Data on the lower diagonal
			if i >= j:
				continue
			else:
				axes[j, i].scatter(df[var1], df[var2], alpha=0.6)

			## KDE/histogram on the diagonal

	## Do regression - could bootstrap
	for i, var1 in enumerate(varlist):
		for j, var2 in enumerate(varlist):
			if i >= j:
				continue

			else:
				## Do the regression
				x = np.array(df[var1]).astype(float)
				y = np.array(df[var2]).astype(float)
				slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)

				axes[j, i].plot(x, intercept + x*slope, c='k')
				text = '$r^{2} = %0.2f$\n$p=%1.0e$' %(r_value**2, p_value)
				axes[i, j].text(0.15, 0.4, text, fontsize=16)


	## Tidy up the plot


	return fig, axes