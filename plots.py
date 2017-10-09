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

"""
Figure prelims
"""
def compare_sample(df, data_dict, varlist, idx):
	fig, axes = plt.subplots(1, figsize=(10,10))
	var = varlist[idx]
	
	def compare(row):
		x = row[var]
		filename = helpers.get_filename(row)
		post = data_dict[filename]
		y = np.mean(post, axis=1)[idx]
		axes.scatter(x, x-y)

	df.apply(compare, axis=1)

	return fig, axes

"""
Figure 1
"""
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

"""
Figure 2
"""
def pairplot(df, var1, var2, ctype='drug_c'):
	fig, axes = plt.subplots(1, figsize=(10,10))

	def plot_pairs(row):
		x = row[var1]
		y = row[var2]
		c = row[ctype]
		ms = row['ms']

		axes.scatter(x, y, c=c, marker=ms)

	df.apply(plot_pairs, axis=1)

	return fig, axes

def trellisplot(df, varlist):
	## Prepare the figure
	num_vars = len(varlist)
	fig, axes = plt.subplots(num_vars, num_vars, figsize=(10,10))

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

	return fig, axes

def univariate_posterior(data_dict, idx, numbins=50):
	fig, axes = plt.subplots(2, 1, figsize=(10,10))

	for key in data_dict.keys():
		dataset = data_dict[key][:, idx]
		data = key.split('_')[:-1]
		c = helpers.get_colour(data)
		print data

		if data[3] == 'D':
			axes[0].hist(dataset, bins=numbins, color=c, histtype='step')

		else:
			axes[1].hist(dataset, bins=numbins, color=c, histtype='step')


	return fig, axes

"""
Figure 3
"""
def IMI_curve(group_dict, num_points=10, num_resamples=10):
	fig, axes = plt.subplots(1, figsize=(10,10))
	k1 = 0.00055

	x0_vals = np.linspace(0, 20, num_points)

	for key in group_dict.keys():
		group_post = group_dict[key]
		data = key.split('_')[:-1]
		c = helpers.get_colour(data)

		predicted_IMIs = []
		for x0 in x0_vals:
			this_x0_predictions = []
			for sample in group_post:
				theta7 = np.power(10., sample[6])
				theta8 = np.power(10., sample[7])

				for repeat in range(num_resamples):
					this_x0_predictions.append(fl.sample_L(x0, k1, theta7, theta8))

			predicted_IMIs.append(np.mean(this_x0_predictions))

		axes.plot(x0_vals, predicted_IMIs, c=c)

	return fig, axes

def IMI_fullness(df, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(10,10))
	k1 = 0.00055

	def fullness_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		g_ends = [] # predictions based on individual posterior means

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				g_ends.append(g_end_feeding)
				true_IMIs.append(p_length)

		## Plot the results
		axes.scatter(g_ends, true_IMIs, c=row['rate_c'], alpha=0.3)

	## Iterate over the dataset
	df.apply(fullness_IMI, axis=1)

	## Axes labels etc
	axes.set_xlabel('Stomach fullness')
	axes.set_ylabel('Observed IMI')

	axes.set_yscale('log')

	return fig, axes

def IMI_prediction(df, num_samples=10, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(10,10))
	k1 = 0.00055

	def predict_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Get PM thetas
		theta7 = np.power(10., row['theta7'])
		theta8 = np.power(10., row['theta8'])

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		indiv_predicts = [] # predictions based on individual posterior means

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				x0 = g_end_feeding
				true_IMIs.append(p_length)

				for i in range(0, num_samples):
					IMI_samples.append(fl.sample_L(x0, k1, theta7, theta8))

				indiv_predicts.append(np.mean(IMI_samples))

		## Plot the results
		#axes.scatter(true_IMIs, indiv_predicts, c=row['drug_c'], alpha=0.1)
		#axes.scatter(true_IMIs, indiv_predicts, c=row['rate_c'], alpha=0.1)
		axes.scatter(true_IMIs, indiv_predicts, c='b', alpha=0.1)

	## Iterate over the dataset
	df.apply(predict_IMI, axis=1)

	## Axes labels etc
	axes.set_xlabel('True IMI')
	axes.set_ylabel('Predicted IMI')

	#axes.set_xlim([0, 5000])
	#axes.set_ylim([0, 5000])

	x = np.linspace(cutoff, 60000, 10)
	axes.plot(x, x, c='k', ls='--')

	axes.set_xscale('log')
	axes.set_yscale('log')

	return fig, axes

def predict_IMI_full_post(df, data_dict, num_resamples=1, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(10,10))
	k1 = 0.00055

	def predict_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		indiv_predicts = [] # predictions based on individual posterior means
		indiv_errs = []

		## Load the posterior sample dictionary
		filename = helpers.get_filename(row)
		post = data_dict[filename]

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				x0 = g_end_feeding
				true_IMIs.append(p_length)

				for i in range(post.shape[1]):
					post_sample = post[:,i]
					theta7 = np.power(10., post_sample[6])
					theta8 = np.power(10., post_sample[7])
					IMI_samples.append(fl.sample_L(x0, k1, theta7, theta8))

				indiv_predicts.append(np.mean(IMI_samples))
				indiv_errs.append(np.std(IMI_samples))

		## Plot the results
		axes.scatter(true_IMIs, indiv_predicts, c=row['drug_c'], marker=row['ms'])
		"""
		axes.errorbar(true_IMIs, 
					  indiv_predicts, 
					  yerr=indiv_errs, 
					  c=row['drug_c'], 
					  fmt='o',
					  marker=row['ms'])
		"""
	## Iterate over the dataset
	df.apply(predict_IMI, axis=1)

	## Axes labels etc
	axes.set_xlabel('True IMI')
	axes.set_ylabel('Predicted IMI')

	axes.set_xlim([0, 20000])
	axes.set_ylim([0, 20000])

	x = np.linspace(0, 60000, 10)
	axes.plot(x, x, c='k', ls='--')

	return fig, axes

def IMI_prediction_KDEmax(df, num_samples=10, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(10,10))
	k1 = 0.00055

	def predict_IMI(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		## Get PM thetas
		theta7 = np.power(10., row['theta7'])
		theta8 = np.power(10., row['theta8'])

		## Iterate through the bouts, getting x0 and IMI once pause length exceeds cutoff
		true_IMIs = []
		indiv_predicts = [] # predictions based on individual posterior means

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				x0 = g_end_feeding
				true_IMIs.append(p_length)

				for i in range(0, num_samples):
					IMI_samples.append(fl.sample_L(x0, k1, theta7, theta8))

				## Make a KDE of the samples and find the maximum value
				x_grid = np.linspace(0, max(IMI_samples), 10000)
				kde = scipy.stats.gaussian_kde(IMI_samples)
				kde_vals = kde.evaluate(x_grid)
				KDEmax = np.argmax(kde_vals)
				#print x_grid[KDEmax]
				indiv_predicts.append(x_grid[KDEmax])

		## Plot the results
		#axes.scatter(true_IMIs, indiv_predicts, c=row['drug_c'], alpha=0.1)
		#axes.scatter(true_IMIs, indiv_predicts, c=row['rate_c'], alpha=0.1)
		axes.scatter(true_IMIs, indiv_predicts, c='b', alpha=0.1)

	## Iterate over the dataset
	df.apply(predict_IMI, axis=1)

	## Axes labels etc
	axes.set_xlabel('True IMI')
	axes.set_ylabel('Predicted IMI')

	#axes.set_xlim([0, 5000])
	#axes.set_ylim([0, 5000])

	x = np.linspace(cutoff, 60000, 10)
	axes.plot(x, x, c='k', ls='--')

	axes.set_xscale('log')
	axes.set_yscale('log')

	return fig, axes

def intake_fullness(df, cutoff=300):
	fig, axes = plt.subplots(1, figsize=(10,10))

	def plot_meals(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)
		
		## Iterate through the bouts, storing meal data once pause length exceeds cutoff
		mealsizes = []
		gut_ends = []
		this_mealsize = 0
		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event

			if p_length > cutoff:
				this_mealsize += 3.5*rate*f_length # add the feeding in this meal
				mealsizes.append(this_mealsize) # store
				this_mealsize = 0 # reset for next meal
				gut_ends.append(g_end_feeding) # store g_end

			else:
				this_mealsize += 3.5*rate*f_length

		c = row['rate_c']
		ms = row['ms']

		axes.scatter(gut_ends, mealsizes, c=c, marker=ms)

	## Apply the plot function across the dataset
	df.apply(plot_meals, axis=1)

	## Set labels etc
	axes.set_xlabel('Stomach fullness at meal termination (kcal)')
	axes.set_ylabel('Meal size (kcal)')


	return fig, axes

"""
Figure 4
"""
def termination_prob(data_dict):
	fig, axes = plt.subplots(1, figsize=(10,10))

	x_vals = np.linspace(0, 20,100)
	for key in data_dict.keys():
		post = data_dict[key]
		theta4 = post[:,3]
		theta5 = post[:,4]
		data = key.split('_')[:-1]
		c = helpers.get_colour(data)


		sig_min, sig_mean, sig_max, sig_int = helpers.get_Q(x_vals, theta4, theta5)

		axes.plot(x_vals, sig_mean, c=c)

	return fig, axes

def param_change_effect(data_dict, indiv, param_idx, delta, num_samples=100, duration=8*60*60):
	fig, axes = plt.subplots(1, figsize=(10,10))

	post = data_dict[indiv]
	post = np.mean(post, axis=0)
	perturbed_params = np.copy(post)
	perturbed_params[param_idx] = perturbed_params[param_idx] + delta
	
	baseline_samples = []
	perturbed_samples = []
	for i in range(num_samples):
		baseline_samples.append(fs.sample(duration, post, 0)[1])
		perturbed_samples.append(fs.sample(duration, perturbed_params, 0)[1])

	axes.hist(baseline_samples, color='b', bins=20)
	axes.hist(perturbed_samples, color='r', bins=20)

	return fig, axes

"""
Figure 5
"""
def dosing_protocol(data_dict, protocol, num_samples=10, cutoff=300):
	## NOTE: currently does not switch posterior through a long pause
	## This should probably be fixed
	fig, axes = plt.subplots(1, figsize=(10,10))

	max_duration = 3600*sum([i[0] for i in protocol])

	## Repeatedly sample from the posteriors sequentially
	all_ts = []
	amounts = []
	for i in range(0, num_samples):
		x0 = 0
		total_amount = 0
		time_series = [[x0]]
		
		for state in protocol:
			duration = state[0]*3600
			post = np.mean(data_dict[state[1]], axis=0)
			x0 = time_series[-1][-1]
			results = fs.sample(duration, post, x0, init_state='F')
			
			time_series.append(results[0])
			total_amount += results[1]
			
			## Use cutoff to check pause state
			events = results[-1]
			if events[-1][3] > cutoff: # would need to redo sampling to get last state otherwise
				final_state = 'L'

			else:
				final_state = 'S'

		time_series = np.hstack(time_series)
		all_ts.append(time_series[:max_duration])
		amounts.append(total_amount)

	## Plot mean time series and bounds
	all_ts = np.array(all_ts)
	xax = np.arange(all_ts.shape[1])
	axes.plot(np.mean(all_ts, axis=0), c='r')
	axes.fill_between(xax,
					  np.percentile(all_ts, 2.5, axis=0), 
					  np.percentile(all_ts, 97.5, axis=0),
					  color='r',
					  alpha=0.3)

	## Plot a sample time series (the last one)
	axes.plot(time_series[:max_duration])

	## Plot the protocol bars
	## TODO: make for general protocols
	protocol1 = np.arange(protocol[0][0]*3600)
	bar = np.zeros(protocol1.shape[0])
	axes.plot(protocol1, bar, c='k')

	print np.mean(amounts), np.std(amounts)

	return fig, axes

def behav_change_effect_group(data_dict, groupname, xmax, num_samples=100, duration=8*60*60):
	fig, axes = plt.subplots(1, figsize=(10,10))

	post = data_dict[groupname]
	post = np.mean(post, axis=0)
	
	baseline_samples = []
	perturbed_samples = []
	for i in range(num_samples):
		baseline_samples.append(fs.sample(duration, post, 0)[1])
		perturbed_samples.append(fs.sample_lim_x(duration, post, 0, xmax)[1])

	axes.hist(baseline_samples, color='b', bins=20, normed=True)
	axes.hist(perturbed_samples, color='r', bins=20, normed=True)

	return fig, axes

def behav_change_effect_indiv(df, xmax, thetas, num_samples=100, duration=8*60*60):
	fig, axes = plt.subplots(8,1, figsize=(10,10))

	def calc_change(row):
		post = row[thetas]

		true_val = row['rate']
		c = row['drug_c']
		ms = row['ms']
		x0 = float(row['x0'])

		samples = []
		for i in range(num_samples):
			samples.append(fs.sample_lim_x(duration, post, x0, xmax)[1])

		delta = 3600.*np.mean(samples)/duration - true_val

		for i in range(0, 8):
			axes[i].scatter(post[i], delta, c=c, marker=ms)

	## Iterate the plotter over the dataframe
	df.apply(calc_change, axis=1)

	return fig, axes
