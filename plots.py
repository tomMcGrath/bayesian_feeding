import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import helpers
import os
import pandas as pd
import scipy

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
