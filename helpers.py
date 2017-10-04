import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pymc3 as pm
import pandas as pd

"""
Data cleaning helper functions
"""
def remove_cancellations(data):
    ## Guard against empty data
    if len(data) == 0:
        return -1, []

    cancel_count = 0
    cleaned_data = []
    for i, event in enumerate(data):
        ## Skip event and add to cancellation count if cancels
        if event[2] == -1.*data[i-1,2]:
            cancel_count += 1

        else:
            cleaned_data.append(event)

    return cancel_count, np.array(cleaned_data)

def remove_negatives(data):
    ## Guard against empty data
    if len(data) == 0:
        return -1, []

    neg_count = 0
    cleaned_data = []
    for i, event in enumerate(data):
        ## Skip event and add to negative count if negative
        if event[2] <= 0:
            neg_count += 1

        else:
            cleaned_data.append(event)

    return neg_count, np.array(cleaned_data)

def remove_outliers(data, amt_max, dur_max, rate_max, dur_min):
    ## Guard against empty data
    if len(data) == 0:
        return -1, []

    outlier_count = 0
    cleaned_data = []
    for i, event in enumerate(data):
        duration = (event[1] - event[0]).total_seconds()
        amt = event[2]
        rate = amt/duration

        if amt > amt_max:
            outlier_count += 1
            continue

        elif duration > dur_max:
            outlier_count += 1
            continue

        elif duration < dur_min:
            outlier_count += 1
            continue

        elif rate > rate_max:
            outlier_count += 1
            continue

        else:
            cleaned_data.append(event)

    return outlier_count, np.array(cleaned_data)

def clean_data(data, amt_max, dur_max, rate_max, dur_min):
    cancel_count, data = remove_cancellations(data)
    neg_count, data = remove_negatives(data)
    outlier_count, data = remove_outliers(data, amt_max, dur_max, rate_max, dur_min)
    return data, cancel_count, neg_count, outlier_count

def filter_data(df, cage_id, start, stop):
    after_start = df[df['start_ts'] >= start].index
    correct_cage = df[df['cage_id'] == cage_id].index
    before_stop = df[df['end_ts'] <= stop].index
    
    full_index = after_start.intersection(correct_cage).intersection(before_stop)
    
    if len(full_index) == 0:
        return [-1]
    
    next_after = pd.Index([full_index[-1] + 1])
    full_index = full_index.union(next_after)
    full_index = full_index.intersection(correct_cage)
    
    return full_index

"""
Functions to read data files and extract empirical data
"""
def get_indiv(trace, idx):
    num_samples = trace.shape[0]
    data = trace[:, idx]

    """    
    data_holder = []
    for i in range(0, num_samples):
        data_holder.append(trace[i, idx])
    
    data = np.stack(data_holder)
    """
    return data

def get_indiv_theta(trace, idx):
    thetas = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8']
    post = []
    for theta in thetas:
        post.append(get_indiv(trace[theta], idx))

    return np.array(post)

def get_dataset(trace, rat_idx):
    thetas = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8']
    post = []
    for theta in thetas:
        post.append(trace[theta][:, rat_idx])

    return np.array(post)
"""
def infer_duration(filename):
    ## NOTE - this is specific to this dataset
    ## If you are using a new dataset or changing the window size, rewrite this function

    drug, dose, recover, period, cage_id, date = filename.split('_')
    date = date.split('.')[0] # strip end of filename

    if drug == 'PYY':
        return 8

    elif drug == 'Lep':
        return 8

    elif drug == 'LiCL':
        return 8

    elif drug == 'saline' and date != '2015-11-23':
        return 8 # normal dataset, not long-duration

    elif drug == 'saline' and date == '2015-11-23' and recover == 'R' and period == 'D':
        return 12

    elif drug == 'saline' and date == '2015-11-23' and recover == 'R' and period == 'L':
        return 12

    elif drug == 'saline' and date == '2015-11-23' and recover == 'N' and period == 'D':
        return 24

    elif drug == 'saline' and date == '2015-11-23' and recover == 'N' and period == 'L':
        return 24

    else:
        print "ERROR parsing filename ", filename
        return 8 # CHANGEME
"""

def infer_duration(filename):
    drug, dose, recover, period, cage_id, duration, date = filename.split('_')
    return float(duration)


def rate_from_file(path, filename):
    dur = infer_duration(filename)
    filepath = path + '/' + filename
    data = np.loadtxt(filepath, delimiter='\t', usecols=(0,1,2,3,4))

    f_lengths = data[:,0]
    g_starts = data[:,1]
    rates = data[:,2]
    p_lengths = data[:,3]
    g_ends = data[:,4]
    
    qty = rates*f_lengths

    #print qty
    #print dur

    return sum(qty)/float(dur)

def amt_from_file(path, filename):
    filepath = path + '/' + filename
    data = np.loadtxt(filepath, delimiter='\t', usecols=(0,1,2,3,4))

    f_lengths = data[:,0]
    g_starts = data[:,1]
    rates = data[:,2]
    p_lengths = data[:,3]
    g_ends = data[:,4]
    
    qty = rates*f_lengths
    return sum(qty)

def cumulative_feeding(filename):
    data = np.loadtxt(filename, delimiter='\t', usecols=(0,1,2,3,4))
    
    ts = [0]
    for i in data:
        ## Append feeding data
        f_length = i[0]
        rate = i[2]
        
        for j in range(int(f_length)):
            ts.append(ts[-1]+rate*3.5)
            
        ## Append waiting data
        p_length = i[3]
        wait = int(p_length)*[ts[-1]]
        ts += wait
    
    return np.array(ts)

def group_c_feeding(folder):
    maxlen = 0
    x = []
    for filename in os.listdir(folder):
        new_ts = cumulative_feeding(folder + filename)
        if len(new_ts) > maxlen:
            maxlen = len(new_ts)
        x.append(np.array(new_ts))

    for i, ts in enumerate(x):
        x[i] = np.pad(ts, (0, maxlen-len(ts)), 'constant', constant_values=(0, np.nan))

    x = np.stack(x)
    return x

def group_amts(folder):
    amts = []
    for filename in os.listdir(folder):
        amts.append(amt_from_file(folder, filename))

    return amts

"""
Trace processing functions
"""
def run_name(tracename):
    return '_'.join(tracename.split('_')[:-1])



"""
Plotting helper functions
"""
def get_colour(data):
    drug, dose, recover, period = data
    dose = float(dose)
    #print drug

    if drug == 'PYY': # graded green
        if dose == 300:
            c = cm.tab20c(0.4)

        elif dose == 7.5:
            c = cm.tab20c(0.45)

        elif dose == 1.5:
            c = cm.tab20c(0.5)

        else:
            print 'ERROR with drug %s, dose %f' %(drug, dose)

    elif drug == 'LiCl': # graded orange
        if dose == 64:
            c = cm.tab20c(0.2)

        elif dose == 32:
            c = cm.tab20c(0.25)

        elif dose == 16:
            c = cm.tab20c(0.3)   
            
        else:
            print 'ERROR with drug %s, dose %f' %(drug, dose)         

    elif drug == 'GLP-1': # graded violet
        if dose == 300:
            c = cm.tab20c(0.6)

        elif dose == 100:
            c = cm.tab20c(0.65)

        elif dose == 30:
            c = cm.tab20c(0.7)   
            
        else:
            print 'ERROR with drug %s, dose %f' %(drug, dose)        

    elif drug == 'sib':
        #c = cm.tab20c(0.6) # yellow
        c= 'y'

    elif drug == 'Ex-4':
        #c = cm.tab20c(0.8) # grey
        c = 'k'

    elif drug == 'Lep':
        c = cm.tab20c(0) # red

    elif drug == 'saline': # graded blue
        if recover == 'A':
            c = cm.tab20c(0.8) # grey

        elif recover == 'R':
            c = cm.tab20c(0) # blue

    elif drug == 'vehicle':
        c = cm.tab20c(0.05)

    else:
        print 'ERROR with drug %s' %(drug)

    return c

# def get_colour(data):
#     drug, dose, recover, period = data

#     """
#     Define colourmaps
#     """
#     ## PYY
#     PYY_dict = {1.5:5, 7.5:8, 300.0:10}
#     PYY_norm = mpl.colors.Normalize(vmin = 0.0,
#                                     vmax = 10)

#     PYY_cmap = cm.Greens

#     PYY_dose_col = cm.ScalarMappable(norm=PYY_norm, cmap=PYY_cmap)
    
#     """
#     Now do the actual mapping
#     """
#     if drug == 'PYY':
#         c = PYY_dose_col.to_rgba((PYY_dict[float(dose)]))
#     elif drug == 'Lep':
#         c = 'r'
#     elif drug == 'LiCL':
#         c = 'y'
#     elif drug == 'saline' and recover == 'N':
#         c = 'k'
#     elif drug == 'saline' and recover == 'R':
#         c = 'b'
#     else:
#         c = 'c' # CHANGEME
#         #c = np.nan
        
#     return c


"""
Posterior helper functions
"""    
def get_indiv(trace, idx):
    num_samples = trace.shape[0]
    
    data_holder = []
    for i in range(0, num_samples):
        data_holder.append(trace[i, idx])
    
    data = np.stack(data_holder)
    return data

def cov_from_chol(num_vars, chol):
    cov = np.zeros((num_vars, num_vars))
    
    for i in range(0, num_vars):        
        chol_factor = chol[i*(i+1)/2:(i+1)*(i+2)/2]
        
        row_to_add = np.zeros(num_vars)
        row_to_add[0:len(chol_factor)] = chol[i*(i+1)/2:(i+1)*(i+2)/2]
        
        cov[i,:] = row_to_add
        
    cov = np.dot(cov, cov.T)
        
    #cov = cov + np.tril(cov).T
    
    #print cov
    
    return cov

def sample_group(trace, group_id, sample_num):
    mu = trace['mu'][sample_num,group_id,:]
    
    chol = trace['chol_cov'+str(group_id)][sample_num,:] # modified to include different cov matrices
    cov = cov_from_chol(8, chol)
    
    return mu, cov

def group_hist(trace, group_list, group_id_dict, theta_idx, recip=False):
    fig, axes = plt.subplots(2,1)

    theta = trace['mu'][:,:,theta_idx]

    for i in group_list:
        theta_var = np.power(10., theta[:,i])
        
        if recip == True:
            theta_var = 1./theta_var

        data = group_id_dict[i].split('_')
        c = get_colour(data)
        
        if data[3] == 'D':
            axes[0].hist(theta_var, label=group_id_dict[i],
                         color=c, alpha=0.6, bins=20, normed=True)
            
        elif data[3] == 'L':
            axes[1].hist(theta_var, label=group_id_dict[i],
                         color=c, alpha=0.6, bins=20, normed=True)  
            
        else:
            raise ValueError
            
    axes[0].legend()
    return fig, axes


"""
Model functions
"""
def get_Q(xvals, theta5, theta6, percentile=5):
    def sig(x, theta5, theta6):
        eps = 0.01
        return eps + (1. - 2*eps)/(1. + np.exp(-0.1*theta5*(x-20*theta6)))

    sigmoids = []
    for i, t5_val in enumerate(theta5):
        sigmoids.append(sig(xvals, t5_val, theta6[i]))

    sigmoids = np.stack(sigmoids, axis=1)

    sig_int = pm.stats.hpd(sigmoids.T)

    sig_min = np.percentile(sigmoids, percentile, axis=1)
    sig_mean = np.mean(sigmoids, axis=1)
    sig_max = np.percentile(sigmoids, 100.-percentile, axis=1)
    
    return sig_min, sig_mean, sig_max, sig_int

"""
Forward sampling helper functions
"""