#!/usr/bin/env python3
'''

Description:

Module for simulating various FMT trial types.  Currently supports greedy donor selection, and even number of patients per donor.

'''

import numpy as np
import scipy
import scipy.stats
import trial
from scipy.stats import beta
import subprocess
import string
from random import shuffle
from optparse import OptionParser
import os

# Read in arguments for the script                                              
usage = "%prog -p PHI_VAL -e EPSILON_VAL -b BETA_VAL"
parser = OptionParser(usage)
parser.add_option("-p", "--phi_val", type="float", dest="phi")
parser.add_option("-e", "--epsilon_val", type="float", dest="epsilon")
parser.add_option("-b", "--beta_val", type="float", dest="beta")
parser.add_option("-N", "--npatients", type="int", dest="npatients")
parser.add_option("-D", "--ndonors", type="int", dest="ndonors")
parser.add_option("-B", "--nblocks", type="int", dest="nblocks")
(options, args) = parser.parse_args()


def simulate_trial_evensplit(npatients_per_donor, donorlist, phi, beta, epsilon):
    # Treatment
    ndonors = len(donorlist)
    ngood_donors = np.sum(donorlist)
    npatients_with_good_donors = npatients_per_donor * ngood_donors
    npatients_with_bad_donors = npatients_per_donor * (ndonors - ngood_donors)
    good_donor_successes = np.random.binomial(npatients_with_good_donors, epsilon)
    bad_donor_successes = np.random.binomial(npatients_with_bad_donors, beta)
    good_donor_failures = npatients_with_good_donors - good_donor_successes
    bad_donor_failures = npatients_with_bad_donors - bad_donor_successes

    # Placebo - number set to be equal to number in treatment arm
    number_of_placebo_patients = npatients_per_donor * len(donorlist)
    placebo_successes = np.random.binomial(number_of_placebo_patients, beta)
    placebo_failures = number_of_placebo_patients - placebo_successes

    # Compute contingency_table                               
    contingency_table = [[good_donor_successes + bad_donor_successes, placebo_successes],
                         [good_donor_failures + bad_donor_failures, placebo_failures]]

    # Evaluate trial outcome
    oddsratio, pval = scipy.stats.fisher_exact(contingency_table)
    nsuccesses = good_donor_successes + bad_donor_successes
    return pval, [contingency_table[0][0], contingency_table[0][1], contingency_table[1][0], contingency_table[1][1]]


def simulate_trial_greedy(npatients, ndonors, phi, epsilon, beta, phi_a, phi_b, epsilon_a, epsilon_b, beta_a, beta_b):
    # Simulate a trial of 'npatients' in each arm (treatment and control), with a given donor list, epsilon and beta values. 
    # Uses greedy donor selection at each step, based on inputed prior shape parameters for phi, beta, epsilon.

    # Setup trial
    random_draws = np.random.uniform(0, 1, npatients)  # draw pseudo-random number sequence associated with patient outcomes
    ngood_donors = np.random.binomial(ndonors, phi)  # draw a number of good donors
    donorlist = ngood_donors * [1] + (ndonors - ngood_donors)*[0]
    shuffle(donorlist)
    state = ndonors*[(0,0)]

    # Setup donor ASCII characters (0 = 'A', 1 = 'B', etc.)
    donor_chars = string.printable[36:36+ndonors]

    # Initialize trial history (e.g. A fails, B succeeds, B fails, C succeeds = AfBsBfCs)
    trial_history = ''

    # Treatment arm
    for patient_idx in range(npatients):
        #print("Simulating patient number " + str(patient_idx))
        # Get posterior predictive probabilities by calling C code
        donor_ppp = []
        sp_args = ["./ppp"]
        sp_args.append(str(phi_a))
        sp_args.append(str(phi_b))
        sp_args.append(str(epsilon_a))
        sp_args.append(str(epsilon_b))
        sp_args.append(str(beta_a))
        sp_args.append(str(beta_b))

        for i, (s, f) in enumerate(state):
            sp_args.append(str(s))
            sp_args.append(str(f))
        with subprocess.Popen(sp_args, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                donor_ppp.append(float(line))

        # Pick donor with highest posterior predictive probability
        max_ppp = np.where(donor_ppp == np.amax(donor_ppp))
        greedy_donor_idx = np.random.choice(max_ppp[0])
        donor_char = donor_chars[greedy_donor_idx]

        # Draw outcomes
        if donorlist[greedy_donor_idx] == 1:
            if random_draws[patient_idx] <= epsilon:
                state = [(s + 1, f) if i == greedy_donor_idx else (s,f) for i, (s,f) in enumerate(state)]
                trial_history += donor_char + 's'
            else:
                state = [(s, f + 1) if i == greedy_donor_idx else (s,f) for i, (s,f) in enumerate(state)]
                trial_history += donor_char + 'f'
        else:
            if random_draws[patient_idx] <= beta:
                state = [(s + 1, f) if i == greedy_donor_idx else (s,f) for i, (s,f) in enumerate(state)]
                trial_history += donor_char + 's'
            else:
                state = [(s, f + 1) if i == greedy_donor_idx else (s,f) for i, (s,f) in enumerate(state)]
                trial_history += donor_char + 'f'

    # Placebo arm
    placebo_successes = np.random.binomial(npatients, beta)
    placebo_failures = npatients - placebo_successes

    
    # Compute contingency_table                               
    treatment_successes = np.sum([s for i, (s,f) in enumerate(state)])
    treatment_failures = np.sum([f for i, (s,f) in enumerate(state)])
    contingency_table = [[treatment_successes, placebo_successes],
                         [treatment_failures, placebo_failures]]

    # Evaluate trial outcome
    oddsratio, pval = scipy.stats.fisher_exact(contingency_table)

    # Is the top ranked donor (most successes) a good donor?
    top_donor_idx = 0
    max_s = 0
    for i, (s,f) in enumerate(state):
        if s > max_s:
            top_donor_idx = i
            max_s = s
    if donorlist[top_donor_idx] == 1:
        top_donor_good = 1
    else:
        top_donor_good = 0

    return pval, state, donorlist, trial_history, [contingency_table[0][0], contingency_table[0][1], contingency_table[1][0], contingency_table[1][1]], top_donor_good


def simulate_trial_block_allocation(npatients, ndonors, nblocks, phi, epsilon, beta, phi_a, phi_b, epsilon_a, epsilon_b, beta_a, beta_b, exponent=1):
    # Simulate a trial of 'npatients' in each arm (treatment and control), with a given donor list, epsilon and beta values. 
    # Performs updates after completion of each block of patients, based on inputed prior shape parameters for phi, beta, epsilon.
    # Also takes as optional input the exponent (currently supports 1 and 2 - for more, update the C code for 'donor_goodness_probabilities') of the donor
    # goodness probability to use for allocation.
    # Allocates at each block based on probability of donor goodness.

    # Setup trial
    random_draws = np.random.uniform(0, 1, npatients)  # draw pseudo-random number sequence associated with patient outcomes
    ngood_donors = np.random.binomial(ndonors, phi)  # draw a number of good donors
    donorlist = ngood_donors * [1] + (ndonors - ngood_donors)*[0]
    shuffle(donorlist)
    state = ndonors*[(0,0)]
    npatients_per_block = int(npatients)/int(nblocks)

    # Setup donor ASCII characters (0 = 'A', 1 = 'B', etc.)
    donor_chars = string.printable[36:36+ndonors]

    # Initialize trial history (e.g. A fails, B succeeds, B fails, C succeeds = AfBsBfCs)
    trial_history = ''

    # Treatment arm, block by block
    for block_idx in range(nblocks):
        # Get posterior goodness probabilities by calling C code
        donor_goodness_probabilities = []
        if exponent == 1:
            sp_args = ["./donor_goodness_probabilities"]
        else:
            sp_args = ["./donor_goodness_probabilities_squared"]
        sp_args.append(str(phi_a))
        sp_args.append(str(phi_b))
        sp_args.append(str(epsilon_a))
        sp_args.append(str(epsilon_b))
        sp_args.append(str(beta_a))
        sp_args.append(str(beta_b))

        for i, (s, f) in enumerate(state):
            sp_args.append(str(s))
            sp_args.append(str(f))
        with subprocess.Popen(sp_args, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                donor_goodness_probabilities.append(float(line))

        # Renormalize to ensure sum to one
        norm_term = np.sum(donor_goodness_probabilities)
        donor_goodness_probabilities = [prob/norm_term for prob in donor_goodness_probabilities] 

        # Allocate patients in block proportionally to the donor goodness probabilities        
        donor_choices = np.random.choice(range(ndonors), size=npatients_per_block, p=donor_goodness_probabilities)
        #donor_char = donor_chars[greedy_donor_idx]

        # Draw outcomes for each patient in block
        patient_ndx = 0
        for donor_ndx in donor_choices:
            if donorlist[donor_ndx] == 1:
                if random_draws[patient_ndx] <= epsilon:
                    state = [(s + 1, f) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 's'
                else:
                    state = [(s, f + 1) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 'f'
            else:
                if random_draws[patient_ndx] <= beta:
                    state = [(s + 1, f) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 's'
                else:
                    state = [(s, f + 1) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 'f'
            patient_ndx += 1

    # Placebo arm
    placebo_successes = np.random.binomial(npatients, beta)
    placebo_failures = npatients - placebo_successes

    
    # Compute contingency_table                               
    treatment_successes = np.sum([s for i, (s,f) in enumerate(state)])
    treatment_failures = np.sum([f for i, (s,f) in enumerate(state)])
    contingency_table = [[treatment_successes, placebo_successes],
                         [treatment_failures, placebo_failures]]

    # Evaluate trial outcome
    oddsratio, pval = scipy.stats.fisher_exact(contingency_table)

    # Is the top ranked donor (most successes) a good donor?
    top_donor_idx = 0
    max_s = 0
    for i, (s,f) in enumerate(state):
        if s > max_s:
            top_donor_idx = i
            max_s = s
    if donorlist[top_donor_idx] == 1:
        top_donor_good = 1
    else:
        top_donor_good = 0

    return pval, state, donorlist, trial_history, [contingency_table[0][0], contingency_table[0][1], contingency_table[1][0], contingency_table[1][1]], top_donor_good



def simulate_trial_greedy_block_allocation(npatients, ndonors, nblocks, phi, epsilon, beta, phi_a, phi_b, epsilon_a, epsilon_b, beta_a, beta_b):
    # Simulate a trial of 'npatients' in each arm (treatment and control), with a given donor list, epsilon and beta values. 
    # Performs updates after completion of each block of patients, based on inputed prior shape parameters for phi, beta, epsilon.
    # Allocates at each block based on greedy choice.

    # Setup trial
    random_draws = np.random.uniform(0, 1, npatients)  # draw pseudo-random number sequence associated with patient outcomes
    ngood_donors = np.random.binomial(ndonors, phi)  # draw a number of good donors
    donorlist = ngood_donors * [1] + (ndonors - ngood_donors)*[0]
    shuffle(donorlist)
    state = ndonors*[(0,0)]
    npatients_per_block = int(int(npatients)/int(nblocks))
    
    # Setup donor ASCII characters (0 = 'A', 1 = 'B', etc.)
    donor_chars = string.printable[36:36+ndonors]

    # Initialize trial history (e.g. A fails, B succeeds, B fails, C succeeds = AfBsBfCs)
    trial_history = ''

    # Treatment arm, block by block
    patient_ndx = 0
    for block_idx in range(nblocks):
        # Get posterior goodness probabilities by calling C code
        donor_ppp = []
        sp_args = ["./ppp"]
        sp_args.append(str(phi_a))
        sp_args.append(str(phi_b))
        sp_args.append(str(epsilon_a))
        sp_args.append(str(epsilon_b))
        sp_args.append(str(beta_a))
        sp_args.append(str(beta_b))

        for i, (s, f) in enumerate(state):
            sp_args.append(str(s))
            sp_args.append(str(f))
        with subprocess.Popen(sp_args, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                donor_ppp.append(float(line))

        # Pick donor with highest posterior predictive probability
        max_ppp = np.where(donor_ppp == np.amax(donor_ppp))
        greedy_donor_idx = np.random.choice(max_ppp[0])
        donor_char = donor_chars[greedy_donor_idx]

        # Allocate patients in block proportionally to the donor goodness probabilities        
        donor_choices = npatients_per_block * [greedy_donor_idx]
        
        # Draw outcomes for each patient in block
        for donor_ndx in donor_choices:
            if donorlist[donor_ndx] == 1:
                if random_draws[patient_ndx] <= epsilon:
                    state = [(s + 1, f) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 's'
                else:
                    state = [(s, f + 1) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 'f'
            else:
                if random_draws[patient_ndx] <= beta:
                    state = [(s + 1, f) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 's'
                else:
                    state = [(s, f + 1) if i == donor_ndx else (s,f) for i, (s,f) in enumerate(state)]
                    trial_history += donor_chars[donor_ndx] + 'f'
            patient_ndx += 1

    # Placebo arm
    placebo_successes = np.random.binomial(npatients, beta)
    placebo_failures = npatients - placebo_successes

    
    # Compute contingency_table                               
    treatment_successes = np.sum([s for i, (s,f) in enumerate(state)])
    treatment_failures = np.sum([f for i, (s,f) in enumerate(state)])
    contingency_table = [[treatment_successes, placebo_successes],
                         [treatment_failures, placebo_failures]]

    # Evaluate trial outcome
    oddsratio, pval = scipy.stats.fisher_exact(contingency_table)

    # Is the top ranked donor (most successes) a good donor?
    top_donor_idx = 0
    max_s = 0
    for i, (s,f) in enumerate(state):
        if s > max_s:
            top_donor_idx = i
            max_s = s
    if donorlist[top_donor_idx] == 1:
        top_donor_good = 1
    else:
        top_donor_good = 0

    return pval, state, donorlist, trial_history, [contingency_table[0][0], contingency_table[0][1], contingency_table[1][0], contingency_table[1][1]], top_donor_good


# Main
phi_a = 1
phi_b = 5
epsilon_a = 7
epsilon_b = 11
beta_a = 2
beta_b = 35

# Parameters
phi = options.phi
beta = options.beta
epsilon = options.epsilon
npatients = options.npatients
ndonors = options.ndonors
nblocks = options.nblocks

# Initialize
Ntrials = 10000
pvals = Ntrials*[0]
trial_histories = Ntrials*['']
top_donors_good = Ntrials*[0]
donorlists = []

with open('contingencies_phi_' + str(phi) + '_eps_' + str(epsilon) + '_beta_' + str(beta) + '_npatients_' + str(npatients) + '_ndonors_' + str(ndonors) + '_nblocks_' + str(nblocks) + '.txt', 'w') as fid_cont:
    fid_cont.write('successes_FMT\tsuccesses_placebo\tfailures_FMT\tfailures_placebo\n')
    for i in range(Ntrials):
        pval, end_state, donorlist, trial_history, contingency_table, top_donor_good = simulate_trial_greedy(npatients, ndonors, nblocks, phi, epsilon, beta, phi_a, phi_b, epsilon_a, epsilon_b, beta_a, beta_b)
        pvals[i] = pval
        trial_histories[i] = trial_history
        top_donors_good[i] = top_donor_good
        donorlists.append(donorlist)
        fid_cont.write('\t'.join([str(int(cont)) for cont in contingency_table]) + '\n')


with open('pvalues_phi_' + str(phi) + '_eps_' + str(epsilon) + '_beta_' + str(beta)  + '_npatients_' + str(npatients) + '_ndonors_' + str(ndonors) + '_nblocks_' + str(nblocks) + '.txt', 'w') as fid:
    for pval in pvals:
        fid.write(str(pval)+'\n')

with open('donorlists_phi_' + str(phi) + '_eps_' + str(epsilon) + '_beta_' + str(beta) + '_npatients_' + str(npatients) + '_ndonors_' + str(ndonors) + '_nblocks_' + str(nblocks) + '.txt', 'w') as fid:
    for donorlist in donorlists:
        fid.write(str(donorlist)+'\n')

with open('histories_phi_' + str(phi) + '_eps_' + str(epsilon) + '_beta_' + str(beta) +  '_npatients_' + str(npatients) + '_ndonors_' + str(ndonors) + '_nblocks_' + str(nblocks) + '.txt', 'w') as fid:
    for history in trial_histories:
        fid.write(str(history)+'\n')



