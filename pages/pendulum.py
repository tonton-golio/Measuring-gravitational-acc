import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from uncertainties import ufloat
from math import pi
from utils import *



def times_func():
	def prepDataFrame():
		df = pd.read_csv("data_project1 - pendul_time.csv", index_col=0, header=[0])
		"""
		###### Exclude some data?
		"""

		cols = st.columns(2)
		chop = cols[0].radio('Chop tail?', [True, False])
		remove_drunk = cols[1].radio('Remove drunk lab-rats?', [True, False])
		if chop:
			df = df[:22]

		cols = st.columns(2)
		times  = df[[f'time_{name}' for name in ['Anton','Adrian','Imke','Majbritt', 'Michael']]]
		times -= times.iloc[0] # subtract initial time
		if remove_drunk:
			times = times.copy().drop(columns=['time_Imke', 'time_Anton'])
		
		times.reset_index(inplace=True)
		times.drop(columns=['number of swings'], inplace=True)
		return times
	def initaxgrid():
		fig = plt.figure(constrained_layout=True,figsize=(12,6))
		gs = GridSpec(2, 3, figure=fig)
		ax = [fig.add_subplot(gs[:, 0]),
				fig.add_subplot(gs[0, 1]),
				fig.add_subplot(gs[1, 1]), 
				fig.add_subplot(gs[:, 2])]
		return fig, ax
	
	times = prepDataFrame()
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
	ax[1].set_yticks([-.2, 0, .2], [-.2, 'mean', .2], )
	ax[1].set(xlabel='swing counter', ylabel='deviation from fit')
		
	
	# lets tally those in a hist
	(counts, bins, _) = ax[2].hist(off_from_mean, bins=20)
	mu_best, sig_best, log_likelihood = maximum_likelihood_finder(counts/max(counts), 
													a_mu = -.5, b_mu =.5, 
													a_sig = 0.1, b_sig = 10.,
													return_plot=False, verbose=False)

													# that didnt quite work...
	
	mu_best, sig_best, A = 0,.1,1.5

	x_plot = np.linspace(min(bins), max(bins), 100)
	ax[2].plot(x_plot, A*norm.pdf(x_plot, scale=sig_best, loc=mu_best))
	ax[2].set(xlabel='Deviation', ylabel='frequency')


	for col in times.columns:
		ax[3].plot(range(len(times)-1),times[col].values[1:] - times[col].values[:-1], label=col)
	ax[3].set(xlabel='Swing Counter', ylabel='Period')
	_ = [i.legend() for i in ax]
	st.pyplot(fig)

	
	st.markdown(f"""
	So we obtain:
	$$
		T = {T:.3f} \pm {sig_best:.3f}
	$$
	""")

	return T, sig_best


def otherMeasurements():
	df = pd.read_csv("data_project1 - pendul_other.csv", index_col=0, header=[1])
	df_err = df[df.columns[1::2]].T
	df = df[df.columns[::2]].T
	df

	fig, ax = plt.subplots(figsize=(12,4))
	df_scaled = df/df.mean(axis=0) #we value which we divide by,  should be moved onto the plot
	df_scaled.boxplot(vert=False)
	mu_round = np.round(df.mean(axis=0), 3)
	std_round = np.round(df.std(axis=0), 3)
	#for i, (m, std) in enumerate(zip(mu_round, std_round)):
	#	plt.text(1.2, 1+i, f'$\mu = {m}, \sigma = {std}$')

	ax.set(xlabel='Value/mean')
	st.pyplot(fig)  # also, can we turn this plot -pi/2


	fig, ax = plt.subplots(1,2, figsize=(6,3))
	L = np.mean(df['Pendulum Length (mm)']) / 1000 # m
	Lerr = np.std(df['Pendulum Length (mm)']) / 1000 # m
	
	
	f"""
	> Should be more clear colors.
	So we obtain:
	$$
		L = {L:.3f} \pm {Lerr:.4f}
	$$

	"""
	return L, Lerr


# Main render script
"""
# pendulum
## data 
### period
> instead measure number of swings. This lets us plot number of swings versus time
"""

T, Terr = times_func()

"### Other Measurements"

L, Lerr =  otherMeasurements()

'# g'
g = L*(2*pi/T)**2

r'''
$$
	g = L\left(\frac{2\pi}{T}\right)^2
$$
> How do we propagate the error to $g$?
'''

L = ufloat(18.728, 0.0005)
T = ufloat(8.640, 0.100)
g_real = 9.81563 #at (55.697039, 12.571243)

def grav_acc(L,T):
    return L*(2*pi/T)**2


g = grav_acc(L,T)

st.markdown(
"""
$$
	g = {:10.2f}
$$
""".format(g))

