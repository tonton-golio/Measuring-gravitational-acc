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

def maximum_likelihood_finder(sample, mus = (-10, 10), sigs = (0.001, 10)):
	'''Uses linear search to find max. likelihood'''
	def log_likelihood( mu=0, sig=1): 
		return np.sum(np.log(norm.pdf(sample, scale=sig, loc=mu)))

	sample = sample[~np.isnan(sample)]
	(mus, sigs) = (np.linspace(*r, 1000) for r in [mus, sigs]) 
	
	# linear search of mus
	estimates_mu = [log_likelihood(mu, sig=1) for mu in mus]
	mu_best = mus[estimates_mu.index(np.max(estimates_mu))]
	
	# linear search of sigmas
	estimates_sig = [log_likelihood(mu_best, sig=sig) for sig in sigs]
	sig_best = sigs[estimates_sig.index(np.max(estimates_sig))]
	
	return mu_best, sig_best, log_likelihood(mu_best, sig_best)
