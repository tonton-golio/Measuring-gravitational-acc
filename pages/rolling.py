import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit, cost
import seaborn as sns
from uncertainties import ufloat
from uncertainties.umath import sin
from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)

def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])

class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2

# Data
filenames = ['measurement1_large_0.csv', 'measurement3_small_180.csv', 
             'measurement1_large_180.csv', 'measurement4_large_0.csv', 
             'measurement1_small_0.csv', 'measurement4_large_180.csv', 
             'measurement1_small_180.csv', 'measurement4_small_0.csv', 
             'measurement2_large_0.csv', 'measurement4_small_180.csv', 
             'measurement2_large_180.csv', 'measurement5_large_0.csv', 
             'measurement2_small_0.csv', 'measurement5_large_180.csv', 
             'measurement2_small_180.csv', 'measurement5_small_0.csv', 
             'measurement3_large_0.csv', 'measurement5_small_180.csv', 
             'measurement3_large_180.csv', 'measurement6_large_180.csv', 
             'measurement3_small_0.csv']

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
    prev = False
    acti = {}
    pnum = -1
    for idx, t in enumerate(df.index):

        V = df['Channel 1 (V)'].iloc[idx]
        if V > 4.758: 
            if prev == False:
                pnum +=1
                acti[pnum] = []
            prev = True
            acti[pnum].append(t)
        else:
            prev = False
                
            
    vals = [np.mean(acti[key]) for key in acti]

    errs = [2*np.std(acti[key]) for key in acti]


    
    if plot:
        cols[0].write(f'Number of peaks found = {len(vals)}')
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        for v, e in zip(vals, errs):
            ax.axvline(v, ls='--', c='r', label='midpoint')
            ax.axvline(v-e, ls='--', c='orange', label=r'midpoint-1 $\sigma$')
            ax.axvline(v+e, ls='--', c='orange', label=r'midpoint+1 $\sigma$')
        dt = vals[1]-vals[0]
        ax.set(xlim=(vals[0]-1*dt/8, vals[0]+1*dt/8))
        #ax.set(xlim=(vals[0]-2*dt, vals[-1]+2*dt))

        #ax.legend()
        cols[1].pyplot(fig)
   
    return vals, errs

def look4many(filenames):
    vals_many = {}
    errs_many = {}
    for filename in filenames[:]:
        vals, errs = look4peaks(filename)
        vals_many[filename[11:-4]] = vals
        errs_many[filename[11:-4]] = errs

    df_vals = pd.DataFrame(vals_many).T
    df_errs = pd.DataFrame(errs_many).T
    
    for df in [df_vals, df_errs]:
        df.reset_index(inplace=True)
        df['size'] = df['index'].apply(lambda x: x.split("_")[1])
        df['flip'] = df['index'].apply(lambda x: x.split("_")[2])
    

    return df_vals, df_errs

def quadratic(x, a, b, c):
    return 1/2*a*x**2. + b*x + c

def extract_values_for_fit(df_vals, df_errs):
    xyyerr_values = {}
    
    for size in ['small', 'large']:
        for flip in [0, 180]:
            df_vals2 = df_vals[ (df_vals['size']==size) & (df_vals['flip']==flip) ]
            df_errs2 = df_errs[ (df_errs['size']==size) & (df_errs['flip']==flip) ]

            X = df_vals2.values[:,1:-2]
            X = (X.T - X[:,0]).T

            Xerr = df_errs2.values[:,1:-2]

            pos = np.array([gate_positions_fmt]*len(X)).T.flatten()
            times = X.T.flatten()
            times_errs = Xerr.T.flatten()
            
            xyyerr_values[size+str(flip)] = (times, pos, times_errs)

    return xyyerr_values


def fit_and_plot(xy_values):
    fig, ax = plt.subplots(1,4, figsize=(12,3))
    i = 0
    fit_vals = {}
    for size in ['small', 'large']:
        for flip in [0, 180]:
            name = size+str(flip)
            (times, pos, times_errs) = xy_values[name]
            ax[i].scatter(times, pos, s=4)

            ax[i].set(title=name, xlabel='time', ylabel='position')
            
            chi2_object_fit = Chi2Regression(quadratic, times, pos, times_errs)
            m = Minuit(chi2_object_fit, a=1.5, b=0.5, c=1.0)
            m.errordef = 1
            m.migrad()
            """
            c = cost.LeastSquares(times, pos, 0.01, quadratic)
            m = Minuit(c, 1.5, 0.5, 1)
            m.migrad()
            """
            

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
            with
            $$
                a = {:10.4f}
            $$
            """.format(size, flip, g, a))

                
############
# render
'# Rolling'
'## Looking for peaks'
cols = st.columns(2)
cols[0].write(f'we have {len(filenames)} files, but lets start with just extracting from one!')
filename = cols[0].selectbox('filename', filenames)
vals, errs = look4peaks(filename, plot=True, cols=cols)


'''Now lets look at all of them'''
#df_vals, df_errs = look4many(filenames); 
#df_vals.to_csv('peaks_vals.csv')
#df_errs.to_csv('peaks_errs.csv')


# load
df_vals = pd.read_csv('peaks_vals.csv', index_col=0)
df_errs = pd.read_csv('peaks_errs.csv', index_col=0)


xyyerr_values = extract_values_for_fit(df_vals, df_errs)

fit_values = fit_and_plot(xyyerr_values)


'#### Results'
calculate_g(fit_values)


