import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import seaborn as sns
from uncertainties import ufloat

from uncertainties.umath import sin  # Imports sin(), etc.
# Data
filenames = '''measurement1_large_0.csv, measurement3_small_180.csv, measurement1_large_180.csv, measurement4_large_0.csv, measurement1_small_0.csv, measurement4_large_180.csv, measurement1_small_180.csv, measurement4_small_0.csv, measurement2_large_0.csv, measurement4_small_180.csv, measurement2_large_180.csv, measurement5_large_0.csv, measurement2_small_0.csv, measurement5_large_180.csv, measurement2_small_180.csv, measurement5_small_0.csv, measurement3_large_0.csv, measurement5_small_180.csv, measurement3_large_180.csv, measurement6_large_180.csv, measurement3_small_0.csv'''.split(', ')[:]

gate_positions = np.array([[176.4528302,	1.509433962],
                    [354.4226415,	1.509433962],	
                    [535.4339623,	1.509433962],	
                    [725.5849057,	1.509433962],	
                    [901.4339623,	1.509433962]])

gate_positions_fmt = (gate_positions[:,0]-gate_positions[0,0])/1000


other_measurements = {
    'theta' : {
        0   : (90-ufloat(75.81624549, 0.04061371841))*np.pi/180,
        180 : (90-ufloat(76.18700361, 0.04061371841))*np.pi/180
    },
    'ball dia' : {
        'small' : ufloat(12.72857143, 0.003571428571)/1000,
        'large' : ufloat(15.01538462, 0.003076923077)/1000,
    },
    'rail sepparation' : ufloat(5.95, 0.00625)/1000    
}



# funcs
def look4peaks(filename, plot=False, cols=[]):
    
    prefix = 'inclline_ball_voltage_measurements/'
    df = pd.read_csv(prefix+filename, encoding='utf-8', header=[13], index_col=0)

    starts = []
    for idx in range(len(df)-1):
        i, j = df['Channel 1 (V)'].iloc[idx], df['Channel 1 (V)'].iloc[idx+1]
        if j-i > 0.2: starts.append(df.index[idx])
    #starts
    true_starts = [starts[0]]
    for i in starts:
        true_starts = np.array(true_starts)
        if (abs(i-true_starts) < 0.1).any():
            pass
        else:
            true_starts = list(true_starts)
            true_starts.append(i)
    
    if plot:
        cols[0].write(f'Number of peaks found = {len(true_starts)}')
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        for i in true_starts:
            ax.axvline(i)
        dt = true_starts[1]-true_starts[0]
        ax.set(xlim=(true_starts[0]-2*dt, true_starts[-1]+2*dt))
        cols[1].pyplot(fig)
   
    return true_starts

def quadratic(x, a, b, c):
        return 1/2*a*x**2 + b*x + c

def look4many(filenames, cols):
    all_starts = {}
    for filename in filenames[:]:
        all_starts[filename[11:-4]] = look4peaks(filename)
    

    def fit_and_plot():
        fit_values = {}

        fig, ax = plt.subplots(2,1, sharex=True, sharey=True, figsize=(4,6))
        

        for (key, vals) in all_starts.items():
        
            if 'large' in key: idx0 = 0
            else: idx0 = 1

            
            try:
                x = (vals-vals[0])
                y =  gate_positions[:,0]/1000
                yerr = gate_positions[:,1]/1000
                ax[idx0].errorbar(x, y, yerr,label=key)
                p0=[2000, 0,y[0]]
                popt, pcov = curve_fit(quadratic, x, y, p0=p0) # use minuit to include yerr
                #x
                x_plot = np.linspace(min(x), max(x), 100)
                ax[idx0].plot(x_plot, quadratic(x_plot, *popt), ls='--', c='r')

                fit_values[key] = popt
                
                
            except: cols[0].write(f'failed on: {key}')


        ax[0].set(title='large ball', xlabel='time (s)', ylabel='gate position (m)')
        ax[1].set(title='small ball', xlabel='time (s)')

        #ax.legend()
        plt.tight_layout()
        cols[1].pyplot(fig)
        return fit_values


    fit_values = fit_and_plot()
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

    def boxes(df2, st=st):
        fig, ax = plt.subplots(3, 1, figsize=(4,6))
        sns.boxplot(data=df2, x='a', y='size', hue='flip',
                        ax=ax[0])
        sns.boxplot(data=df2, x='b', y='size', hue='flip', 
                        ax=ax[1])
        sns.boxplot(data=df2, x='c', y='size', hue='flip',
                    ax=ax[2])
        plt.tight_layout()
        st.pyplot(fig)
    boxes(df2, st=cols[0])
    cols[0].table(df2)
    return df2


def look4many(filenames, cols):
    all_starts = {}
    for filename in filenames[:]:
        all_starts[filename[11:-4]] = look4peaks(filename)

    df = pd.DataFrame(all_starts).T
    df.reset_index(inplace=True)
    df['size'] = df['index'].apply(lambda x: x.split("_")[1])
    df['flip'] = df['index'].apply(lambda x: x.split("_")[2])

    for i in range(1,5):
        df[i]=df[i]-df[0]
    df[0] = 0

    return df


def next(df):
    #df_mini = df.groupby(['size', 'flip']).mean().T
    #gate_positions_fmt
    #df_mini.set_index(gate_positions_fmt, inplace=True)
    
    #df_mini
    fit_values = {}
    fig, ax = plt.subplots(2,2)
    for i, size in enumerate(['small', 'large']):
        for j, flip in enumerate([0, 180]):
            df2 = df[ (df['size']==size) & (df['flip']==flip) ]
            
            X = df2.values[:,1:-2]
            X = X.copy()
            mu = np.mean(X, axis=0)

            std = [np.std(x) for x in X.T]
            
            ax[i,j].errorbar(mu, gate_positions_fmt, xerr=std, yerr=1.509433962/1000)

            p0=[2000, 0,1]
            popt, pcov = curve_fit(quadratic, mu, gate_positions_fmt, p0=p0) # use minuit to include yerr
            #x
            x_plot = np.linspace(min(mu), max(mu), 100)
            ax[i,j].plot(x_plot, quadratic(x_plot, *popt), ls='--', c='r')
            
            fit_values[size+str(flip)] = popt

    st.pyplot(fig)

    return fit_values


                
############
# render
'# Rolling'
'## Looking for peaks'
cols = st.columns(2)
cols[0].write(f'we have {len(filenames)} files, but lets start with just extracting from one!')
# file selector:
filename = cols[0].selectbox('filename', filenames)


true_starts = look4peaks(filename, plot=True, cols=cols)


'''Now lets look at all of them'''

cols = st.columns(2)
#df = look4many(filenames, cols); df.to_csv('peaks.csv')

df = pd.read_csv('peaks.csv', index_col=0)
fit_values = next(df)

'fit_values:', fit_values


'#### Results'
for size in ['small', 'large']:
    for flip in [0, 180]:
        a = fit_values[size+str(flip)][0]
        
        theta = other_measurements['theta'][flip]
        D = other_measurements['ball dia'][size]
        d = other_measurements['rail sepparation']
        theta, D, d

        g = (a / sin(theta)) * (1 +  2/5 * D**2/ (D**2 - d**2))

        st.markdown("""
        {}, {} yields:
        $$
            g = {:10.4f}
        $$
        """.format(size, flip, g))

