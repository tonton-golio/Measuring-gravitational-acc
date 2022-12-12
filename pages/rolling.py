import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit, cost
import seaborn as sns
from uncertainties import ufloat
from uncertainties.umath import sin


# Data
filenames = '''measurement1_large_0.csv, measurement3_small_180.csv, measurement1_large_180.csv, measurement4_large_0.csv, measurement1_small_0.csv, measurement4_large_180.csv, measurement1_small_180.csv, measurement4_small_0.csv, measurement2_large_0.csv, measurement4_small_180.csv, measurement2_large_180.csv, measurement5_large_0.csv, measurement2_small_0.csv, measurement5_large_180.csv, measurement2_small_180.csv, measurement5_small_0.csv, measurement3_large_0.csv, measurement5_small_180.csv, measurement3_large_180.csv, measurement6_large_180.csv, measurement3_small_0.csv'''.split(', ')[:]

gate_positions = np.array([[176.4528302,	1.509433962],
                    [354.4226415,	1.509433962],	
                    [535.4339623,	1.509433962],	
                    [725.5849057,	1.509433962],	
                    [901.4339623,	1.509433962]
                    ])

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

def quadratic(x, a, b, c):
        return 1/2*a*x**2. + b*x + c

def extract_values_for_fit(df):
    fit_values = {}
    
    for size in ['small', 'large']:
        for flip in [0, 180]:
            df2 = df[ (df['size']==size) & (df['flip']==flip) ]
            X = df2.values[:,1:-2]

            pos = np.array([gate_positions_fmt]*len(X)).T.flatten()
            times = X.T.flatten()
            
            fit_values[size+str(flip)] = (times, pos)

    return fit_values

def fit_and_plot(xy_values):
    fig, ax = plt.subplots(1,4, figsize=(12,3))
    i = 0
    fit_vals = {}
    for size in ['small', 'large']:
        for flip in [0, 180]:
            name = size+str(flip)
            (times, pos) = xy_values[name]
            ax[i].scatter(times, pos, s=4)

            ax[i].set(title=name, xlabel='time', ylabel='position')

            c = cost.LeastSquares(times, pos, 0.01, quadratic)
            m = Minuit(c, 1.5, 0.5, 1)
            m.migrad()

            vals = [m.values[p] for p in 'a b c'.split()]
            errs = [m.errors[p] for p in 'a b c'.split()]


            x_plot = np.linspace(min(times), max(times), 100)
            ax[i].plot(x_plot, quadratic(x_plot, *vals))
            i+=1

            fit_vals[name] = (vals, errs)


    plt.tight_layout()
    st.pyplot(fig)

    return fit_vals

def calculate_g(fit_values):

    for size in ['small', 'large']:
        for flip in [0, 180]:
            name = size+str(flip)
            a = fit_values[name][0][0]
            a_err = fit_values[name][1][0]

            a = ufloat(a, a_err)
            
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
#df = look4many(filenames, st); df.to_csv('peaks.csv')
df = pd.read_csv('peaks.csv', index_col=0)


xy_values = extract_values_for_fit(df)


fit_values = fit_and_plot(xy_values)


'#### Results'
calculate_g(fit_values)


