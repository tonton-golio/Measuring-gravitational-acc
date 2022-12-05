from utils import *

"""
# pendulum
## data 
### period
> instead measure number of swings. This lets us plot number of swings versus time
"""

"""
## Cut of tail?
Some fuckery occurs at the end, show we just chop it off? 
"""


df = pd.read_csv("data_project1 - pendul_time.csv", index_col=0, header=[0])

chop = st.button('chop', )
#
# 
# if chop:
df = df[:25]


cols = st.columns(2)
times  =df[['time_Anton',	
			'time_Adrian',	
			'time_Imke',
			'time_Majbritt',	
			'time_Michael']] # view
times -= times.iloc[0]
#times.drop(columns=['time_Anton'], inplace=True)#
times.reset_index(inplace=True)
times.drop(columns=['number of swings'], inplace=True)
#times
# should we just take the mean and std like this?
mean_time = times.mean(axis=1) # we have the 
# same uncertainties for now, so this is fine
std_time = times.std(axis=1)


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


fig = plt.figure(constrained_layout=True,figsize=(12,6))

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[:, 0])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])




ax = [ax1, ax2, ax3]
#ax[0].errorbar(times.index, mean_time,std_time)




x = np.array(list(times.index))
y = mean_time


popt, pcov = curve_fit(linear_0Bound, x, y, p0=[1])  # origin bound linear fit, slope is T
# minuit
T = popt[0]
Td = y-x*T


X = np.vstack([x]*len(times.columns)).T
off_from_mean = (times-X*T).values.flatten()

for t in times.columns:
	ax[0].scatter(times.index, times[t]-x*T, marker='x', s=50, label=t.split('_')[1])
	

mu_best, sig_best, log_likelihood = maximum_likelihood_finder(Td, 
												a_mu = -4, b_mu =4, 
												a_sig = .1, b_sig = 10.,
												return_plot=False, verbose=False)



#x_plot = np.linspace(min(x), max(x)+5, 100)
#ax[0].plot(x_plot, linear_0Bound(x_plot, *popt), label=f'f=({round(popt[0], 3)}$\pm?$)x')
#ax[0].legend()
#ax[0].set(title = "Period = $t/N$", ylabel = "time, $t$", xlabel = "number of swings, $N$", 
#            xlim = (0,max(x)*1.1), ylim = (0,max(y)*1.1)) 


(counts, bins, _) = ax[1].hist(off_from_mean, bins=20)
x_plot = np.linspace(min(bins)*2, max(bins)*2, 100)
ax[1].plot(x_plot, len(off_from_mean)**2*norm.pdf(x_plot, scale=sig_best, loc=mu_best))

ax[0].legend()
st.pyplot(fig)

st.markdown(f"""
So we obtain:
$$
	T = {T:.3f} \pm {sig_best:.3f}
$$
""")


"""
### Length
"""

Ls = 6 + np.random.randn(200)
pd.DataFrame(Ls, columns=['Length']).T
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].boxplot(Ls)
ax[1].hist(Ls)


L, Lerr, log_likelihood = maximum_likelihood_finder(Td, a_mu = 6, b_mu =12, a_sig = 0.0001, b_sig = 10, return_plot=False, verbose=False)

x_plot = np.linspace(min(Ls), max(Ls), 100)
plt.plot(x_plot, 100*norm.pdf(x_plot, L, Lerr, ))

st.pyplot(fig)


# remember to propagate this error through


f"""
So we obtain:
$$
	T = {L:.3f} \pm {Lerr:.3f}
$$

"""


'# g'

g = L*(2*pi/T)**2
r'''
$$
	g = L\left(\frac{2\pi}{T}\right)^2
$$'''

f"""
$$
	g = {g:.3f} \pm {1.2:.3f}
$$
"""

