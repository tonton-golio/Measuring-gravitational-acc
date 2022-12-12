import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from uncertainties import ufloat, unumpy
from math import pi
from utils import *
import pandas as pd
from scipy.optimize import curve_fit
from statistics import stdev

def lin_func(x,a,b):
    return a*x + b

def grav_acc(L,T):
    return L*(2*pi/T)**2

def length_of_pend(pend_other_data):
    L_pend = unumpy.uarray(pend_other_data.iloc[0].to_numpy()[1::2],
                           pend_other_data.iloc[0].to_numpy()[2::2])
    L_pend_mean = np.mean(L_pend)
    print('Pendulum length (m):', L_pend_mean/1000)
    mass = unumpy.uarray(pend_other_data.iloc[1].to_numpy()[1::2],
                         pend_other_data.iloc[1].to_numpy()[2::2])
    mass_mean = np.mean(mass)
    L_hook = unumpy.uarray(pend_other_data.iloc[2].to_numpy()[1::2],
                           pend_other_data.iloc[2].to_numpy()[2::2])
    L_hook_mean = np.mean(L_hook)
    L_mass_ground = unumpy.uarray(pend_other_data.iloc[3].to_numpy()[1::2],
                                  pend_other_data.iloc[3].to_numpy()[2::2])
    L_mass_ground_mean = np.mean(L_mass_ground)
    print('Distance Pendulum Ground (m):',L_mass_ground_mean/100)
    L_mass_without_hook = unumpy.uarray(pend_other_data.iloc[4].to_numpy()[1::2],
                                     pend_other_data.iloc[4].to_numpy()[2::2])
    L_mass_without_hook_mean = np.mean(L_mass_without_hook)
    print('Width of Pendulum (without hook) (m):',L_mass_without_hook_mean)
    L_mass_with_hook = unumpy.uarray(pend_other_data.iloc[5].to_numpy()[1::2],
                                        pend_other_data.iloc[5].to_numpy()[2::2])
    L_mass_with_hook_mean = np.mean(L_mass_with_hook)
    L_pend_tape = unumpy.uarray(pend_other_data.iloc[6].to_numpy()[1::2],
                                pend_other_data.iloc[6].to_numpy()[2::2])
    L_pend_tape_mean = np.mean(L_pend_tape)
    print('Distance Pendulum Ground with tape measure (m):',L_pend_tape_mean/1000)
    
    return (L_pend_mean-L_mass_ground_mean*10-L_mass_without_hook_mean*10/2)/1000

def get_periods(pend_time):
    
    def plot_swings_vs_period(name = False, include_Imke_Anton = False):
        fig, ax = plt.subplots(dpi = 400)
        ax.scatter(n_swings_Adrian, time_Adrian, label = 'Adrian')
        popt, pcov = curve_fit(lin_func,n_swings_Adrian, time_Adrian)
        ax.plot(n_swings_Adrian,lin_func(n_swings_Adrian, *popt))
        ax.scatter(n_swings_Majbritt, time_Majbritt, label = 'Majbritt')
        popt, pcov = curve_fit(lin_func,n_swings_Majbritt, time_Majbritt)
        ax.plot(n_swings_Majbritt,lin_func(n_swings_Majbritt, *popt))
        ax.scatter(n_swings_Michael, time_Michael, label = 'Michael')
        popt, pcov = curve_fit(lin_func,n_swings_Michael, time_Michael)
        ax.plot(n_swings_Michael,lin_func(n_swings_Michael, *popt))
        if include_Imke_Anton == True:
            ax.scatter(n_swings_Anton, time_Anton, label = 'Anton')
            popt, pcov = curve_fit(lin_func,n_swings_Anton, time_Anton)
            ax.plot(n_swings_Anton,lin_func(n_swings_Anton, *popt))
            ax.scatter(n_swings_Imke, time_Imke, label = 'Imke')
            popt, pcov = curve_fit(lin_func,n_swings_Imke, time_Imke)
            ax.plot(n_swings_Imke,lin_func(n_swings_Imke, *popt))
        ax.set_ylabel(r'Period $T$ in s')
        ax.set_xlabel(r'$n$ Swings')
        ax.legend()
        ax.grid()
        if name != False:
            fig.savefig(name, dpi = 400)
        return True
    
    n_swings = pend_time.iloc[:,0].to_numpy()
    
    time_Anton = pend_time.iloc[:,1].to_numpy()
    time_Anton = time_Anton[~np.isnan(time_Anton)]
    time_Anton = time_Anton[1:] - time_Anton[0:-1]
    n_swings_Anton = n_swings[:len(time_Anton)]
    time_Adrian = pend_time.iloc[:,2].to_numpy()
    time_Adrian = time_Adrian[~np.isnan(time_Adrian)]
    time_Adrian = time_Adrian[1:] - time_Adrian[0:-1]
    n_swings_Adrian = n_swings[:len(time_Adrian)]
    time_Imke = pend_time.iloc[:,3].to_numpy()
    time_Imke = time_Imke[~np.isnan(time_Imke)]
    time_Imke = time_Imke[1:] - time_Imke[0:-1]
    n_swings_Imke = n_swings[:len(time_Imke)]
    time_Majbritt = pend_time.iloc[:,4].to_numpy()
    time_Majbritt = time_Majbritt[~np.isnan(time_Majbritt)]
    time_Majbritt = time_Majbritt[1:] - time_Majbritt[0:-1]
    n_swings_Majbritt = n_swings[:len(time_Majbritt)]
    time_Michael = pend_time.iloc[:,5].to_numpy()
    time_Michael = time_Michael[~np.isnan(time_Michael)]
    time_Michael = time_Michael[1:] - time_Michael[0:-1]
    n_swings_Michael = n_swings[:len(time_Michael)]

    #plot_swings_vs_period()
    
    time_Adrian = np.delete(time_Adrian, [21,22])
    n_swings_Adrian = n_swings[:len(time_Adrian)]
    time_Michael = time_Michael[:-4]
    n_swings_Michael = n_swings[:len(time_Michael)]
    
    plot_swings_vs_period('period_vs_swings.pdf', include_Imke_Anton = True)
    
    plot_swings_vs_period('period_vs_swings.pdf')

    period_Anton = ufloat(np.mean(time_Anton),stdev(time_Anton))
    period_Adrian = ufloat(np.mean(time_Adrian),stdev(time_Adrian))
    period_Imke = ufloat(np.mean(time_Imke),stdev(time_Imke))
    period_Majbritt = ufloat(np.mean(time_Majbritt),stdev(time_Majbritt))
    period_Michael = ufloat(np.mean(time_Michael),stdev(time_Michael))
    
    periods = [period_Adrian, period_Majbritt, period_Michael, 
               period_Imke, period_Anton]
    names = ['Adrian', 'Majbritt', 'Michael', 'Imke', 'Anton']
    return names, periods

def times_func():
	def prepDataFrame():
		# load 
		df = pd.read_csv("data_project1 - pendul_time.csv", index_col=0, header=[0])
		
		# display options
		cols = st.columns(3)
		cols[0].markdown("###### Exclude some data?")
		chop = cols[1].radio('Chop tail?', [True, False])
		remove_drunk = cols[2].radio('Remove drunk lab-rats?', [True, False])
		A = 2
		if chop: df = df[:22]
		if remove_drunk: df = df.copy().drop(columns=['time_Imke', 'time_Anton',
													'terr_Imke', 'terr_Anton'])
		else:A=75
		
		# extract times
		times  = df[df.columns[:len(df.columns)//2]]
		times -= times.iloc[0] # subtract initial time
		times.reset_index(inplace=True)
		times.drop(columns=['number of swings'], inplace=True)
		
		return times, A
	def initaxgrid():
		fig = plt.figure(constrained_layout=True,figsize=(12,6))
		gs = GridSpec(2, 3, figure=fig)
		ax = [fig.add_subplot(gs[:, 0]),
				fig.add_subplot(gs[0, 1]),
				fig.add_subplot(gs[1, 1]), 
				fig.add_subplot(gs[:, 2])]
		return fig, ax
	

	times, A = prepDataFrame()
	x = np.array(list(times.index))
	y = mean_time = times.mean(axis=1)  # same uncertainties so for now; this is fine
	yerr = times.std(axis=1)
	
	popt, pcov = curve_fit(linear_0Bound, x, y, p0=[10])  # origin bound linear fit, slope is T. # use minuit instead
	T = popt[0] # period

	fig, ax = initaxgrid()
	ax[0].errorbar(x, y,yerr, lw=0, elinewidth=0, capsize=1, label='errorbars')
	ax[0].plot(x, linear_0Bound(x, T), c='r', ls='--', label=f'fit: {round(T,4)}x')
	ax[0].set(xlabel='swing counter', ylabel='time (s)')

	# how far are all the points away from the line
	X = np.vstack([x]*len(times.columns)).T
	off_from_mean = (times-X*T).values.flatten()
	for t in times.columns:
		ax[1].scatter(times.index, times[t]-x*T, marker='x', s=50, label=t.split('_')[1])
	ax[1].set(xlabel='swing counter', ylabel='deviation from fit')
		
	
	# lets tally those in a hist
	(counts, bins, _) = ax[2].hist(off_from_mean, bins=25)
	
	mu_best, sig_best, _ = maximum_likelihood_finder(off_from_mean)
	

	x_plot = np.linspace(min(off_from_mean), max(off_from_mean), 100)
	ax[2].plot(x_plot, A*norm.pdf(x_plot, scale=sig_best, loc=mu_best), 
				label = f'normal, {round(mu_best,3)}, {round(sig_best,3)}')
	ax[2].set(xlabel='Deviation', ylabel='frequency')


	for col in times.columns:
		ax[3].plot(range(len(times)-1),times[col].values[1:] - times[col].values[:-1], label=col)
	ax[3].set(xlabel='Swing Counter', ylabel='Period')
	_ = [i.legend() for i in ax]
	plt.close()

	T = ufloat(T, sig_best)
	return T, fig

def otherMeasurements(): # L
	df_orig = pd.read_csv("data_project1 - pendul_other.csv", index_col=0, header=[1])
	df_err = df_orig[df_orig.columns[1::2]].T
	df = df_orig.copy()[df_orig.columns[::2]].T
	
	
	w = weigted_dict = {}
	for col in df.columns:
		mean_weighted = sum(df[col].values * df_err[col].values)/ df_err[col].sum()
		std_weighted = np.sqrt(1/sum(1/df_err[col].values))
		w[col] = ufloat(mean_weighted, std_weighted)
		
	def plotBox():
		fig, ax = plt.subplots(figsize=(12,4))
		df_scaled = df/df.mean(axis=0)
		df_scaled.boxplot(vert=False)

		for i, col in enumerate(df.columns):
			rounded = np.round([w[col].n,w[col].s],3)
			plt.text(1.0085, .6+i, 
								# weighted averages
								r'$\hat{\mu} = $' + str(rounded[0]) +\
								r'  $\hat{\sigma} = $' + str(rounded[1])+\
								# mean and std
								'\n$\mu = {}, \sigma = {}$'.format(*np.round([df[col].mean(),df[col].std()], 3)))

		ax.set(xlabel='Value/mean')
		plt.close()
		return fig
	
	fig = plotBox()

	return weigted_dict, df_orig, fig

def return_L_T_g():
    pend_other_data = pd.read_csv('data_project1 - pendul_other.csv',
                                  skiprows=[0])
    pend_time = pd.read_csv('data_project1 - pendul_time.csv')
    
    names, periods = get_periods(pend_time)
    
    print(names)
    print(periods)
    
    #period including Anton and Imke
    #T = (period_Anton + period_Adrian + period_Imke + period_Majbritt + period_Michael)/5
    #print(T)
    
    #period without Anton and Imke
    T = np.sum(periods[:3])/3
    print(T)
    L = length_of_pend(pend_other_data)
    
    print(grav_acc(L,T))
    return L,T,grav_acc(L,T)

def main(): # Main render script
	r"""
	# Pendulum
	Gravitational acc. is related to a pendulum's length and period by;
	$$
		g = L\left(\frac{2\pi}{T}\right)^2
	$$
	"""
    
	L,T,g = return_L_T_g()
    
	############ T
	"##### Period"

	T2, fig = times_func()
	st.pyplot(fig)

	st.markdown(
	"""
	$$
		T = {:10.2f}
	$$
	""".format(T))


	############ L
	"##### Length"

	weigted_dict, df, fig =  otherMeasurements()
	L2 = weigted_dict['Pendulum Length (all made by adrian in mm)']/1000 # m

	with st.expander('raw', expanded=False):
		df
	st.pyplot(fig) # Should be clearer colors and Rotate bars

	st.markdown("""
	If we shouldn't comebine the numbers from the box_plot, we get:
	$$
		L = {:10.4f}
	$$
	""".format(L))


	############ g
	"##### Gravitational acc."
	#L = ufloat(18.728, 0.0005) # i think this uncertainty is too small
	#T = ufloat(8.640, 0.100)

	g_real = 9.81563 #at (55.697039, 12.571243)

	st.markdown(
	"""
	$$
		g = {:10.2f}
	$$
	""".format(g))

main()