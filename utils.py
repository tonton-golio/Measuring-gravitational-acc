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

def golden_section_min(f,a,b,tolerance=1e-7, maxitr=1e3):
    factor = (np.sqrt(5)+1)/2

    # define c and d, which are points between a and b
    n_calls = 0
    while (abs(a-b)>tolerance) and (maxitr > n_calls):
        c = b - (b-a)/factor
        d = a + (b-a)/factor

        if f(c) > f(d):
            a = c
        else:
            b=d
        #st.write(f(a),f(c),f(d),f(b))
        n_calls += 1
    return (c+d)/2, n_calls

def maximum_likelihood_finder(sample, a_mu = -2, b_mu =2, a_sig = 0.5, b_sig = 0.5, return_plot=False, verbose=False):

	xs = np.linspace(min(sample), max(sample), 100)

	# determine likelihood  (should really implement bisection)
	log_likelihood = lambda mu=0, sig=1, a=1: a*np.sum(np.log(norm.pdf(sample, scale=sig, loc=mu)))


	# golden_section_min
	
	
	start_golden_search = time()
	f_mu_neg = lambda mu:-1*log_likelihood(mu, sig=1, a=1)
	mu_best, ncalls_mu = golden_section_min(f_mu_neg,a_mu,b_mu,tolerance=1e-5, maxitr=1e3)

	f_sig_neg = lambda sig:-1*log_likelihood(mu_best, sig =1)
	sig_best, ncalls_sig = golden_section_min(f_sig_neg,a_sig,b_sig,tolerance=1e-5, maxitr=1e3)
	stop_golden_search = time()


	
	if verbose:
		cols = st.columns(2)
		num_round = 4
		cols[1].write(f"""
			golden_section_min: 

				mu_best = {round(mu_best, num_round)}
				sig_best = {round(sig_best, num_round)}
				ncalls_mu =  {ncalls_mu}
				ncalls_sig =  {ncalls_sig}
				time = {round(stop_golden_search-start_golden_search,3)}
			""")

	def plot():
		fig, ax = plt.subplots(1,2, figsize=(8,3))
		ax[0].hist(sample, density=True, label='sample', 
					bins=int(len(sample)**.9)//3,
					color='red',
					alpha=.5)
		ys = norm.pdf(xs, scale=sig_best, loc=mu_best)
		ax[0].plot(xs, ys, label='best', color='cyan', ls='--')
		ax[0].set_xlabel('sample value', color='beige')
		ax[0].set_ylabel('occurance frequency', color='beige')
		#ax[1].set_yscale('log')

		for i in ax:
			l = i.legend(fontsize=12)
			for text in l.get_texts():
				text.set_color("white")


		plt.tight_layout()
		plt.close()
		return fig
	if return_plot: return mu_best, sig_best, log_likelihood(mu_best, sig_best), plot()
	else: return mu_best, sig_best,  log_likelihood(mu_best, sig_best)
