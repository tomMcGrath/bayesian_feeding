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
import itertools

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
def ts_from_data(data):
	k1 = 0.00055
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

	return bout_ts, stomach_ts

def timeseries_predict_plot(data, thetas, predict_index, num_samples=100, tmax=60*60):
	k1 = 0.00055
	theta7 = np.power(10., thetas[6])
	theta8 = np.power(10., thetas[7])
	fig, axes = plt.subplots(2,1, figsize=(10,10))

	bout_ts, stomach_ts = ts_from_data(data)

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
	ax2.hist(next_times, histtype='step', color='r', bins=100, normed=True)

	## KDE of results
	"""
	x_grid = np.arange(len(stomach_ts))
	kde = scipy.stats.gaussian_kde(next_times, bw_method='silverman')
	y = kde.evaluate(x_grid)
	ax2.plot(x_grid[int(start_time):], y[int(start_time):], c='r')
	ax2.fill_between(x_grid[int(start_time):], 0, y[int(start_time):], color='r', alpha=0.3)
	ax2.set_ylim([0, 2.5*np.max(y)])
	ax2.set_yticklabels([])
	"""
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

	plt.subplots_adjust(hspace=0.1)

	return fig, axes

def plot_ethogram(folder, num_animals, maxlen=None, downsample=10, ysize=50):
	num_animals = min(num_animals, len(os.listdir(folder)))

	fig, axes = plt.subplots(2*num_animals, 1, figsize=(10,10))

	for i, filename in enumerate(os.listdir(folder)):
		if i >= num_animals:
			continue

		filepath = folder + '/' + filename
		data = np.loadtxt(filepath, delimiter='\t', usecols=(0,1,2,3,4))
		bout_ts, stomach_ts = ts_from_data(data)

		## Set the feeding to binary and downsample
		bout_ts = bout_ts[:maxlen] > 0.0
		bout_ts = bout_ts[::downsample]

		## Downsample the stomach fullness
		stomach_ts = stomach_ts[:maxlen]
		stomach_ts = stomach_ts[::downsample]

		## Make rasterplot of bouts
		x = np.outer(np.ones(ysize), bout_ts)
		axes[2*i].matshow(x, cmap=cm.Greys)
		## Plot stomach fullness
		y = np.outer(np.ones(ysize), stomach_ts)
		axes[2*i + 1].matshow(y, cmap=cm.YlOrRd)

	for i in range(2*num_animals):
		axes[i].set_xticklabels([])
		axes[i].set_yticklabels([])
		axes[i].set_yticks([])

	#fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.001)
	plt.subplots_adjust(hspace=0.05)
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
	fig, axes = plt.subplots(3, 1, figsize=(10,10))
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
		residuals = []
		next_sizes = []
		for i, event in enumerate(data):
			f_length, g_start, rate, p_length, g_end_feeding = event

			IMI_samples = []
			if p_length > cutoff:
				x0 = g_end_feeding
				true_IMIs.append(p_length)

				for j in range(0, num_samples):
					IMI_samples.append(fl.sample_L(x0, k1, theta7, theta8))

				mean_predict = np.mean(IMI_samples)
				indiv_predicts.append(mean_predict)
				resid = p_length - mean_predict
				residuals.append(resid)

				if i < len(data)-1:
					next_size = 3.5*data[i+1][0]*data[i+1][2]
					next_sizes.append(next_size)

		## Plot the results
		#axes.scatter(true_IMIs, indiv_predicts, c=row['drug_c'], alpha=0.1)
		#axes.scatter(true_IMIs, indiv_predicts, c=row['rate_c'], alpha=0.1)
		axes[0].scatter(true_IMIs, indiv_predicts, c='b', alpha=0.1)
		axes[1].scatter(true_IMIs, residuals, c='b', alpha=0.1)
		axes[2].scatter(residuals[:len(next_sizes)], next_sizes, c='b', alpha=0.025)

	## Iterate over the dataset
	df.apply(predict_IMI, axis=1)

	## Axes labels etc
	axes[0].set_xlabel('True IMI')
	axes[0].set_ylabel('Predicted IMI')

	#axes.set_xlim([0, 20000])
	#axes.set_ylim([0, 20000])

	## Visual guides
	x = np.linspace(cutoff, 8*3600, 10)
	axes[0].plot(x, x, c='k', ls='--')
	axes[1].axhline(0, c='k', ls='--')

	axes[0].set_xscale('log')
	axes[0].set_yscale('log')
	axes[1].set_xscale('log')

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

	#axes.set_xlim([0, 20000])
	#axes.set_ylim([0, 20000])

	x = np.linspace(cutoff, 8*3600, 10)
	axes.plot(x, x, c='k', ls='--')

	axes.set_xscale('log')
	axes.set_yscale('log')

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

	x = np.linspace(cutoff, 8*3600, 10)
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

def fullness_IMI(df, cutoff=300, exp_param=1.5):
	fig, axes = plt.subplots(2, 1, figsize=(10,10))

	p_lengths = []
	def plot_data(row):
		## Import the meal data for this animal
		data = helpers.data_from_row(row)

		for event in data:
			f_length, g_start, rate, p_length, g_end_feeding = event
			
			#p_lengths.append(np.log10(p_length))
			if p_length < cutoff:
				p_lengths.append(p_length)

			axes[0].scatter(g_end_feeding, p_length, c='b', alpha=0.1)

	df.apply(plot_data, axis=1)

	lambd = np.power(10., exp_param)
	samples = np.random.exponential(lambd, size=1000)
	#samples = np.log10(samples)

	axes[0].axhline(cutoff, c='k', ls='--')
	axes[0].set_yscale('log')
	axes[1].hist([samples, p_lengths], bins=10, normed=True, color=['b', 'r'])
	#axes[1].hist(p_lengths, bins=20, normed=True)
	#axes[1].axvline(np.log10(cutoff), c='k', ls='--')
	axes[1].axvline(cutoff, c='k', ls='--')

	return fig, axes

def plot_IMI(data_dir, data_dict, cutoff=300, num_samples=10):
    windowsize = 5
    err_thresh = 20000
    
    p_lengths = []
    g_ends = []
    c = []

    ## Assemble dataset
    count = len(os.listdir(data_dir))
    for i, dataset in enumerate(os.listdir(data_dir)):
        data = np.loadtxt(data_dir+dataset, delimiter='\t', usecols=(0,1,2,3,4))

        for j in data:
            f_lengths, g_starts, rates, p_length, g_end = j

            if p_length > cutoff and p_length < err_thresh:
                p_lengths.append(p_length)
                g_ends.append(g_end)
                c.append(float(i)/count)

            if p_length > err_thresh:
                print 'Error in %s, IMI of %3.0f' %(dataset, p_length)

    ## Process to use in moving window
    results = zip(g_ends, p_lengths)
    results = np.array(results)

    ## Generate moving window mean
    x_grid = np.linspace(0, np.max(results[:,0]), 100)

    means = []
    for xval in x_grid:
        usevals = np.abs(results[:,0] - xval) < windowsize
        meanval = np.mean(results[usevals, 1])
        means.append(meanval)

    """
    for i in results:
        print '%3.3f\t%3.3f' %(i[0], i[1])
    """
    means = np.array(means)

    ## Plot moving window
    fig, axes = plt.subplots(1)
    axes.scatter(results[:,0], 
                 results[:,1], 
                 alpha=0.3, 
                 c='b', # use c=c to get individual colour coding
                 cmap=cm.gist_ncar)
    
    axes.plot(x_grid, means, c='k')
    
    ## Generate posterior predictive curve
    post = data_dict[data_dir.split('/')[1]+'_trace.p']
    post = np.mean(post, axis=0)
    
    mean_ppcs = []
    ppcs_low = []
    ppcs_high = []
    k1 = 0.00055
    sample_x_grid = np.linspace(0, np.max(results[:,0]), 30)
    for xval in sample_x_grid:
        samples = []
        for i in range(num_samples):
            theta7 = np.power(10., post[6])
            theta8 = np.power(10., post[7])
            samples.append(fl.sample_L(xval, k1, theta7, theta8))
            
        mean_ppcs.append(np.mean(samples))
        ppc_low = np.percentile(samples, 5)
        ppc_high = np.percentile(samples, 95)
        ppcs_low.append(ppc_low)
        ppcs_high.append(ppc_high)
        
    axes.plot(sample_x_grid, mean_ppcs, c='r')
    axes.fill_between(sample_x_grid, ppcs_low, ppcs_high, color='r', alpha=0.3)
    
    plt.show()


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
	fig, axes = plt.subplots(1, figsize=(10,10))

	## Do the sampling
	amounts, all_ts, last_ts = helpers.sample_protocol(data_dict, protocol, num_samples, cutoff)

	## Plot mean time series and bounds
	xax = np.arange(all_ts.shape[1])
	axes.plot(np.mean(all_ts, axis=0), c='r')
	axes.fill_between(xax,
					  np.percentile(all_ts, 2.5, axis=0), 
					  np.percentile(all_ts, 97.5, axis=0),
					  color='r',
					  alpha=0.3)

	## Plot a sample time series (the last one)
	axes.plot(last_ts)

	## Plot the protocol bars
	## TODO: make for general protocols
	protocol1 = np.arange(protocol[0][0]*3600)
	bar = np.zeros(protocol1.shape[0])
	axes.plot(protocol1, bar, c='k')

	print np.mean(amounts), np.std(amounts)

	return fig, axes

def optimise_protocols(data_dict, druglist, protocol_size, duration, min_default=2, num_samples=10, cutoff=300, pc=5):
	## NOTE: default drug is the first one in the list
	fig, axes = plt.subplots(3, 1, figsize=(10,10))

	## Create protocol list
	protocol_list = helpers.make_protocol_list(druglist, protocol_size, duration, min_default)

	## Iterate over protocols
	mean_amounts = []
	all_amounts = []
	time_series = []
	for protocol in protocol_list:
		amounts, ts_mean, ts_pc, last_ts = helpers.sample_protocol(data_dict, protocol, num_samples, cutoff, 5)
		mean_amounts.append(np.mean(amounts))
		time_series.append((ts_mean, ts_pc))
		all_amounts.append(amounts)

	## Plot policy ranking
	protocols_to_rank = zip(mean_amounts, protocol_list)
	ranked_protocols = sorted(protocols_to_rank, key=lambda x:x[0])
	xax = np.arange(len(mean_amounts))
	amounts_to_plot = [i[0] for i in ranked_protocols] # extract amount eaten
	axes[0].scatter(xax, amounts_to_plot) # could do errorbar for SEM if necessary/useful

	## Plot optimal & pessimal protocol stomach fullness
	protocols_to_rank = zip(mean_amounts, time_series)
	ranked_protocols = sorted(protocols_to_rank, key=lambda x:x[0])
	opt_protocol = ranked_protocols[0]
	pess_protocol = ranked_protocols[-1]

	## This is a bit hacky but works OK
	opt_mean = opt_protocol[1][0]
	opt_pc = opt_protocol[1][1]
	opt_pc_low = opt_pc[0]
	opt_pc_high = opt_pc[1]

	pess_mean = pess_protocol[1][0]
	pess_pc = pess_protocol[1][1]
	pess_pc_low = pess_pc[0]
	pess_pc_high = pess_pc[1]
	
	xax = np.arange(len(opt_mean))

	axes[1].plot(opt_mean, c='b')
	axes[1].fill_between(xax,
				  		 opt_pc_low, 
				  		 opt_pc_high,
				  		 color='b',
				  		 alpha=0.3)

	axes[1].plot(pess_mean, c='r')
	axes[1].fill_between(xax,
			  		 	 pess_pc_low, 
			  			 pess_pc_high,
			  			 color='r',
			  			 alpha=0.3)

	## Plot amount distributions for optimal & pessimal protocols
	protocols_to_rank = zip(mean_amounts, all_amounts)
	ranked_protocols = sorted(protocols_to_rank, key=lambda x:x[0])
	opt_protocol = ranked_protocols[0]
	pess_protocol = ranked_protocols[-1]
	axes[2].hist(opt_protocol[1], color='b', bins=20, normed=True)
	axes[2].hist(pess_protocol[1], color='r', bins=20, normed=True)

	return fig, axes, ranked_protocols

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