import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from uncertainties import ufloat
from math import pi
from utils import *


def times_func():
	def prepDataFrame():
		df = pd.read_csv("data_project1 - pendul_time.csv", index_col=0, header=[0])
		

		cols = st.columns(3)
		cols[0].markdown("###### Exclude some data?")
		chop = cols[1].radio('Chop tail?', [True, False])
		remove_drunk = cols[2].radio('Remove drunk lab-rats?', [True, False])
		if chop:
			df = df[:22]


		cols = st.columns(2)
		times  = df[[f'time_{name}' for name in ['Anton','Adrian','Imke','Majbritt', 'Michael']]]
		times -= times.iloc[0] # subtract initial time
		if remove_drunk:
			times = times.copy().drop(columns=['time_Imke', 'time_Anton'])
			A=2
		else:
			A=75
		
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
	#ax[1].set_yticks([-.2, 0, .2], [-.2, 'mean', .2], )
	ax[1].set(xlabel='swing counter', ylabel='deviation from fit')
		
	
	# lets tally those in a hist
	(counts, bins, _) = ax[2].hist(off_from_mean, bins=25)
	
	mu_best, sig_best, _ = maximum_likelihood_finder(off_from_mean)
	
	
	

	x_plot = np.linspace(min(off_from_mean), max(off_from_mean), 100)
	ax[2].plot(x_plot, A*norm.pdf(x_plot, scale=sig_best, loc=mu_best), label=f'normal, {round(mu_best,3)}, {round(sig_best,3)}')
	ax[2].set(xlabel='Deviation', ylabel='frequency')


	for col in times.columns:
		ax[3].plot(range(len(times)-1),times[col].values[1:] - times[col].values[:-1], label=col)
	ax[3].set(xlabel='Swing Counter', ylabel='Period')
	_ = [i.legend() for i in ax]
	plt.close()

	
	T = ufloat(T, sig_best)

	return T, fig

def otherMeasurements():
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

def main():
	# Main render script
	r"""
	# Pendulum
	Gravitational acc. is related to a pendulum's length and period by;
	$$
		g = L\left(\frac{2\pi}{T}\right)^2
	$$
	"""


	############ T
	"##### Period"

	T, fig = times_func()
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
	L = weigted_dict['Pendulum Length (all made by adrian in mm)']/1000 # m

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

	def grav_acc(L,T):
		return L*(2*pi/T)**2

	g_real = 9.81563 #at (55.697039, 12.571243)
	g = grav_acc(L,T)

	st.markdown(
	"""
	$$
		g = {:10.2f}
	$$
	""".format(g))


main()