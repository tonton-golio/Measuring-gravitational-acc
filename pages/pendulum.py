from utils import *

"""
# pendulum
## data 
### period
> instead measure number of swings. This lets us plot number of swings versus time
"""

df = pd.read_csv("data_project1 - pendul_time.csv", index_col=0, header=[0])
cols = st.columns(2)
cols[0].table(df.T)#.T # view

x = np.hstack(list(df.index)*5)
y = []
yerr = []

fig, ax = plt.subplots()

for t_col, terr_col in zip(df.columns[::2], df.columns[1:][::2]):
	who = t_col.split('_')[1]
	ax.errorbar(df.index, df[t_col], 10*df[terr_col], marker='x',
          ms=2, lw=0, elinewidth=1,capsize=1, label=who)
	y.append(df[t_col].values)
	yerr.append(df[terr_col].values)
ax.legend()

# now lets take the weighted mean at each number of swings


y = np.array(y).flatten()
yerr = np.array(yerr).flatten()


cols[1].pyplot(fig)


popt, pcov = curve_fit(linear_0Bound, x, y, p0=[1])  # origin bound linear fit, slope is T

T = popt[0]
Td = y-x*T

mu_best, sig_best, log_likelihood = maximum_likelihood_finder(Td, a_mu = -1, b_mu =1, a_sig = 0.5, b_sig = 5, return_plot=False, verbose=False)


fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].scatter(x, y)
x_plot = np.linspace(min(x), max(x)+5, 100)
ax[0].plot(x_plot, linear_0Bound(x_plot, *popt), label=f'f=({round(popt[0], 3)}$\pm?$)x')
ax[0].legend()
ax[0].set(title = "Period = $t/N$", ylabel = "time, $t$", xlabel = "number of swings, $N$", 
            xlim = (0,max(x)*1.1), ylim = (0,max(y)*1.1)) 


(counts, bins, _) = ax[1].hist(Td, bins=20)
x_plot = np.linspace(min(bins)*2, max(bins)*2, 100)
plt.plot(x_plot, len(x)**.5*norm.pdf(x_plot, scale=sig_best, loc=mu_best))
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


L, Lerr, log_likelihood = maximum_likelihood_finder(Td, a_mu = 6, b_mu =12, a_sig = 0.0001, b_sig = 1, return_plot=False, verbose=False)

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

