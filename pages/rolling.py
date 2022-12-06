import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'# Rolling'
'## Looking for peaks'


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
for filename in filenames:
    all_starts[filename[10:]] = look4peaks(filename)

fig, ax = plt.subplots()
for key in all_starts:
    vals = all_starts[key]
    ax.plot(vals-vals[0], label=key)

ax.legend()
st.pyplot(fig)