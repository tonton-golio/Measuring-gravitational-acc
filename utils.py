import streamlit as st
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import norm
from time import time


def linear_0Bound(x, a):
    return a*x

def golden_section_min(f,a,b,tolerance=1e-7, maxitr=1e1):
    factor = (np.sqrt(5)+1)/2

    # define c and d, which are points between a and b
    n_calls = 0
    #st.write(dict(zip([a,b], [f(a),f(b)])))
    while (abs(a-b)>tolerance) and (maxitr > n_calls):
        c = b - (b-a)/factor
        d = a + (b-a)/factor
        
        if f(c) > f(d):
            a = c
        else:
            b=d
        
        
        n_calls += 1
    return (c+d)/2, n_calls

def maximum_likelihood_finder(sample, mus = (-10, 10), 
										sigs = (0.001, 10)):

	xs = np.linspace(min(sample), max(sample), 100)
	sample = sample[~np.isnan(sample)]
	# determine likelihood  (should really implement bisection)
	log_likelihood = lambda mu=0, sig=1: np.sum(np.log(norm.pdf(sample, scale=sig, loc=mu)))


	
	(mus, sigs) = (np.linspace(*r, 1000) for r in [mus, sigs]) 
	
	estimates_mu = [log_likelihood(mu, sig=1) for mu in mus]
	mu_best = mus[estimates_mu.index(np.max(estimates_mu))]
	
	estimates_sig = [log_likelihood(mu_best, sig=sig) for sig in sigs]
	sig_best = sigs[estimates_sig.index(np.max(estimates_sig))]
	


	return mu_best, sig_best, log_likelihood(mu_best, sig_best)
