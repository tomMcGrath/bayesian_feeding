import sys
import os
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import PDMP_ll as ll
import cPickle as pickle
import sys

num_samples = int(sys.argv[1])
num_tune = int(sys.argv[2])
num_chains = int(sys.argv[3])
target_accept = float(sys.argv[4])
max_treedepth = int(sys.argv[5])
run_name = sys.argv[6]

#theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'

"""
Data import
"""
min_num_animals = 5

root_data_dir = 'new_all_data/'
group_paths = os.listdir(root_data_dir)

def data_from_file(filename):
    data = np.loadtxt(filename,delimiter='\t',usecols=(0,1,2,3,4))

    f_lengths = data[:,0]
    g_starts = data[:,1]
    rates = data[:,2]
    p_lengths = data[:,3]
    g_ends = data[:,4]

    obs = [f_lengths, g_starts, rates, p_lengths, g_ends]

    return obs

## Iterate over groups
data_holder = []
idx_holder = []
group_idx_holder = []
group_sizes = []
count = 0

subj_data = []

for group_num, path in enumerate(group_paths):
    group_path = root_data_dir+path
    
    if len(os.listdir(group_path)) >= min_num_animals:
        group_sizes.append(len(os.listdir(group_path)))
        #print "Using ", group_path
        ## Iterate over members of group
        for filename in os.listdir(group_path):
            if filename [-3:] != 'CSV':
                continue
                
            #print filename

            new_data = data_from_file(group_path+'/'+filename)
            data_holder.append(new_data)

            ## Create index holders
            idx_holder.append(count*np.ones(len(new_data[0])))
            group_idx_holder.append(group_num*np.ones(len(new_data[0])))
            
            ## Store subject data
            subj_data.append([group_path, filename])

            count += 1
        
    else:
        print "Skipping ", group_path

data = np.hstack(data_holder)
idx = np.hstack(idx_holder)
group_idx = np.hstack(group_idx_holder)
idx = idx.astype(int)
group_idx = group_idx.astype(int)

num_animals = len(data_holder)
num_groups = np.max(group_idx) + 1

print "Number of data points: ", len(group_idx)
print "Number of animals: ", num_animals
print "Number of groups: ", num_groups

## Set constants
k1 = 0.00055

## Model setup
with pm.Model() as model:
    ## Group mean
    #means = [-2, -2.35, -2.8, 1, 1, -2, 3.5, 2] # from unpooled data, all vars
    means = [-2, -2, -3, 1, 1, -2, 3, 3] # from unpooled data, all vars
    #means = [-2, -2.35, -2.8] # from unpooled data, theta1 to theta4
    #means = [-1, -2, -2, 3.5, 2] # from unpooled data, theta5 to theta9
    #means = [-1, -1, -1] # test how strongly priors affect posterior
    num_vars = len(means)
    cov = np.eye(num_vars)

    mu = pm.Normal('mu', mu=means, sd=2, shape=(num_groups,num_vars))

    #mu_print = theano.printing.Print('mu')(mu)

    theta_holder = []
    for i in range(0, num_groups):
        ## Create covariance matrix from LKJ
        #sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=num_vars)
        sd_dist = pm.HalfNormal.dist(sd=2, shape=num_vars)
        packed_chol = pm.LKJCholeskyCov('chol_cov'+str(i), eta=1, n=num_vars, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(num_vars, packed_chol, lower=True)

        theta_tilde = pm.Normal('theta_tilde'+str(i), mu=0, sd=1, shape=(group_sizes[i], num_vars))
        theta_gp = tt.dot(chol, theta_tilde.T).T # have to take transpose to fit with distribution shape
        theta_holder.append(theta_gp)

    thetas = tt.concatenate(theta_holder)

    theta1 = pm.Deterministic('theta1', mu[group_idx,0] + thetas[idx,0]) 
    theta2 = pm.Deterministic('theta2', mu[group_idx,1] + thetas[idx,1]) 
    theta3 = pm.Deterministic('theta3', mu[group_idx,2] + thetas[idx,2]) 
    theta4 = pm.Deterministic('theta4', mu[group_idx,3] + thetas[idx,3]) 
    theta5 = pm.Deterministic('theta5', mu[group_idx,4] + thetas[idx,4]) 
    theta6 = pm.Deterministic('theta6', mu[group_idx,5] + thetas[idx,5]) 
    theta7 = pm.Deterministic('theta7', mu[group_idx,6] + thetas[idx,6]) 
    theta8 = pm.Deterministic('theta8', mu[group_idx,7] + thetas[idx,7])
    
    """
    Power-transform
    """
    p10_theta1 = tt.pow(10., theta1)
    p10_theta2 = tt.pow(10., theta2)
    p10_theta3 = tt.pow(10., theta3)
    p10_theta6 = tt.pow(10., theta6)
    p10_theta7 = tt.pow(10., theta7)
    p10_theta8 = tt.pow(10., theta8)

    ## Likelihood of observations

    ## Exponential feeding bout length
    feeding_lengths = pm.Exponential('f_len', p10_theta1, observed=data[0,:])

    ## Normal feeding bout rate
    rates = pm.Normal('rate', p10_theta2, sd=p10_theta3, observed=data[2,:])

    ## Pause likelihood
    pauses = ll.pause_ll('pause', theta4, theta5, p10_theta6, p10_theta7, p10_theta8, k1, observed=data)

    ## Checking out different step methods to see which works
    # NUTS w/o ADVI - currently fails on LKJCholeskyCov
    trace = pm.sample(num_samples, tune=num_tune, njobs=num_chains,
                      step=pm.NUTS(), target_accept=target_accept, max_treedepth=max_treedepth)
    #trace = pm.sample(num_samples)
    
pickle.dump(trace, open(run_name+"_trace.p", "wb"))
pickle.dump(subj_data, open(run_name+"_subj.p", "wb"))
pickle.dump(group_paths, open(run_name+"_paths.p", "wb"))