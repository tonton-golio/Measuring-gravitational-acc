import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
'# Rolling'


filenames = '''measurement1_large_0.csv, measurement3_small_180.csv, measurement1_large_180.csv, measurement4_large_0.csv, measurement1_small_0.csv, measurement4_large_180.csv, measurement1_small_180.csv, measurement4_small_0.csv, measurement2_large_0.csv, measurement4_small_180.csv, measurement2_large_180.csv, measurement5_large_0.csv, measurement2_small_0.csv, measurement5_large_180.csv, measurement2_small_180.csv, measurement5_small_0.csv, measurement3_large_0.csv, measurement5_small_180.csv, measurement3_large_180.csv, measurement6_large_180.csv, measurement3_small_0.csv'''.split(', ')

st.write(f'we have {len(filenames)} files')

"But lets start with just extracting from one!"
filename = filenames[0]
prefix = 'inclline_ball_voltage_measurements/'
df = pd.read_csv(prefix+filename, encoding='utf-8', header=[13], index_col=0)


fig, ax = plt.subplots()
df.plot(ax=ax)
st.pyplot(fig)


