import numpy as np
from scipy import integrate
import sys
from datetime import datetime, timedelta
import csv
import CLAMS_parsers as parser
import ODEs as odes

"""
Read in the data and digestion rate
"""
def parse_file(infile, d_rate, wanted_period, hours_from, hours_to):
	bout_data = []
	with open(infile, 'r') as data:
		for line in data:
			line = line.strip('\n').split('\t')
			start = datetime.strptime(line[0], "%d/%m/%Y %H:%M:%S")
			end = datetime.strptime(line[1], "%d/%m/%Y %H:%M:%S")
			amount = float(line[2])

			bout_data.append([start, end, amount])


	exp_start = bout_data[0][0]

	"""
	Now do the integrating
	"""
	events = []
	g_start = 0.0 # initialise to 0.0     
	for i, bout in enumerate(bout_data):
		#print bout
		if i != len(bout_data) - 1:
			"""
			Record feeding -> pause -> feeding cycle (this is what we need
			to assign to short or long pause)
			Format:
			(feeding length, stomach at start of feeding, feeding rate,
			pause length, stomach at start of pause)
			"""
			t_start = bout[0]
			t_end = bout[1]
			f_length = (t_end - t_start).total_seconds()
			t_next = bout_data[i+1][0]
			p_length = (t_next - t_end).total_seconds()
			d_start = 6
			d_end = 18

			if (t_start.hour < d_start or t_start.hour >= d_end):
				period = 'D' # dark period
			else:
				period = 'L' # light period
			
			feeding_interval = np.linspace(0, f_length, f_length) # want at 1s resolution 
			pause_interval = np.linspace(0, p_length, p_length) # as above
			
			rate = float(bout[2])/f_length
			
			"""
			Now solve ODE on feeding
			"""
			g_feeding = integrate.odeint(odes.feeding_ode, g_start, feeding_interval, args=(rate,))
			g_feeding = g_feeding.clip(min=0) # remove the small numerical errors bring it < 0
			g_end_feeding = g_feeding[:,0][-1]
			
			g_digestion = integrate.odeint(odes.digestion_ode, g_end_feeding, pause_interval, args=(d_rate,))
			g_digestion = g_digestion.clip(min=0) # remove the small numerical errors bring it < 0
			g_end = g_digestion[:,0][-1]

			"""
			Store time from start for conditional export
			"""
			t_from_start = float((t_start - exp_start).total_seconds())/3600.
			
			"""
			Append
			"""
			event = [f_length, g_start, rate, p_length, g_end_feeding, period, t_from_start]
			events.append(event)
			
			"""
			Setup for the next go round the loop
			"""
			g_start = g_end

	"""
	Output the bout data
	"""
	outputs_data = []
	for event in events:
		print event
		if event[5] == wanted_period and event[6] > hours_from and event[6] <= hours_to:
			outputs_data.append(event)

	return outputs_data