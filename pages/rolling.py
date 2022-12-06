import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
'# Rolling'
'## Looking for peaks'
import seaborn as sns


filenames = '''measurement1_large_0.csv, measurement3_small_180.csv, measurement1_large_180.csv, measurement4_large_0.csv, measurement1_small_0.csv, measurement4_large_180.csv, measurement1_small_180.csv, measurement4_small_0.csv, measurement2_large_0.csv, measurement4_small_180.csv, measurement2_large_180.csv, measurement5_large_0.csv, measurement2_small_0.csv, measurement5_large_180.csv, measurement2_small_180.csv, measurement5_small_0.csv, measurement3_large_0.csv, measurement5_small_180.csv, measurement3_large_180.csv, measurement6_large_180.csv, measurement3_small_0.csv'''.split(', ')

st.write(f'we have {len(filenames)} files, but lets start with just extracting from one!')
# file selector:
filename = st.selectbox('filename', filenames)


def look4peaks(filename, plot=False):
    
    prefix = 'inclline_ball_voltage_measurements/'
    df = pd.read_csv(prefix+filename, encoding='utf-8', header=[13], index_col=0)

    starts = []
    for idx in range(len(df)-1):
        i, j = df['Channel 1 (V)'].iloc[idx], df['Channel 1 (V)'].iloc[idx+1]
        if j-i > 0.3: starts.append(df.index[idx])

    true_starts = [starts[0]]
    for i in starts:
        true_starts = np.array(true_starts)
        if (abs(i-true_starts) < 0.1).any():
            pass
        else:
            true_starts = list(true_starts)
            true_starts.append(i)
    
    if plot:
        st.write(f'Number of peaks found = {len(true_starts)}')
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        for i in true_starts:
            ax.axvline(i)

        st.pyplot(fig)
   
    return true_starts

look4peaks(filename, plot=True)


'''Now lets look at all of them'''

all_starts = {}
for filename in filenames[:]:
    all_starts[filename[11:-4]] = look4peaks(filename)

gate_positions = np.array([[176.4528302,	1.509433962],
                    [354.4226415,	1.509433962],	
                    [535.4339623,	1.509433962],	
                    [725.5849057,	1.509433962],	
                    [901.4339623,	1.509433962]])
#gate_positions

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

fit_values = {}

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
all_starts

for key in all_starts:

    if 'large' in key:
        idx0 = 0
    else:
        idx0 = 1
    vals = all_starts[key]
    try:
        x = vals-vals[0]
        y =  gate_positions[:,0]
        yerr = gate_positions[:,1]
        ax[idx0].errorbar(x, y, yerr,label=key)
        p0=[2000, 0,y[0]]
        popt, pcov = curve_fit(quadratic, x, y, p0=p0) # use minuit to include yerr
        #x
        x_plot = np.linspace(min(x), max(x), 100)
        ax[idx0].plot(x_plot, quadratic(x_plot, *popt), ls='--', c='r')

        fit_values[key] = popt
        
        
    except:
        st.write(f'failed on: {key}')


ax[0].set(title='large ball', xlabel='time (s)', ylabel='gate position (mm)')
ax[1].set(title='small ball', xlabel='time (s)')

#ax.legend()
st.pyplot(fig)



df2 = pd.DataFrame(fit_values, index=['a', 'b', 'c']).T
df2.reset_index(inplace=True)

def flipped(x):
    if '180' in x:
        return True
    else:
        return False

def large(x):
    if 'large' in x:
        return 'large'
    else:
        return 'small'

df2['flip'] = df2['index'].apply(lambda x: flipped(x))
df2['size'] = df2['index'].apply(lambda x: large(x))
df2.drop(columns='index', inplace=True)



df2


fig, ax = plt.subplots(1, 3, figsize=(12,4))
sns.boxplot(data=df2, x='a', y='size', hue='flip',
                 ax=ax[0])
sns.boxplot(data=df2, x='b', y='size', hue='flip', 
                ax=ax[1])
sns.boxplot(data=df2, x='c', y='size', hue='flip',
             ax=ax[2])

st.pyplot(fig)


a_small = df2[(df2["size"]=='large') & (df2['flip']==False)].a.mean()

a = a_small /1000 # m/s^2
theta = 90-13.99837545 # deg
theta *= np.pi/180 # rad

d_theta = 0.3014429894 # deg
d_theta *= np.pi/180 # rad

R = 15.01538462 /1000 # m
d = 5.95 / 1000 # rail spacing  (m)

g = a / np.sin(theta+d_theta) * (1+2/5 * R**2/ (R**2 - (d/2)**2))

g