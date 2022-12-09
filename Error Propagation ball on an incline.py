#Error propagation for a ball on an incline experiment
import sys
sys.path.append('C:/Users/micha/OneDrive - University of Copenhagen/Kode/Applied Statistics/GitHub/External_Functions')
import numpy as np
from sympy import *
from iminuit import Minuit    
import matplotlib.pyplot as plt
from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax 

def incline(x, y, z, q):
    return ( x / np.sin( y * np.pi / 180 ) ) * ( 1 + ( 2/5 ) * ( z**2 ) / ( - q**2 + z**2 ) )

def ballandrail(z, q):
    return ( 1 + ( 2/5 ) * ( z**2 ) / ( - q**2 + z**2 ) )

def inversesinus(y):
    return 1 / np.sin( y * np.pi / 180 )

def pdf_normalized(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mu)**2 / sigma**2)

def pdf_extended(x, N, mu, sigma) :
    return N * pdf_normalized(x, mu, sigma)

#Setting values
#a:x, theta:y, ball:z, rail:q
#Acceleration fit and uncertainty
x, dx = 1.6, 0.0001              #[m/s^2]
#Angle of table and uncertainty
#Setup 000deg
y, dy = 14.00, 0.08         #[Deg]
#Setup 180deg
#y, dy = 13.70, 0.11        #[Deg]
#Big ball
z, dz = 15.015, 0.003       #[mm]
#Small ball
#z, dz = 12.728, 0.004      #[mm]
#Length of rail and uncertainty
q, dq = 5.950, 0.006        #[mm]


#____________________________ Monte Carlo Method ______________________________
r = np.random
r.seed(42)
Npoints=100000

#Simulating data
xgauss = r.normal(loc=x, scale=dx, size=Npoints)
ygauss = r.normal(loc=y, scale=dy, size=Npoints)
zgauss = r.normal(loc=z, scale=dz, size=Npoints)
qgauss = r.normal(loc=q, scale=dq, size=Npoints)

nbins = 100

#Evaluating function
g = [incline(xgauss[i], ygauss[i], zgauss[i], qgauss[i]) for i in range(Npoints)]

#Fitting gaussian with Minuit
binned_likelihood = BinnedLH(pdf_extended, 
                             g,
                             bins=nbins, 
                             bound=(0, 20),
                             extended=True)
Minuit.print_level = 1
minuit = Minuit(binned_likelihood, mu=9.75, sigma=0.05, N=Npoints) 
minuit.errordef = Minuit.LIKELIHOOD
minuit.migrad() #Initiate minuit with migrad

fit_N, fit_mu, fit_sigma = minuit.values[:]     
for name in minuit.parameters:
    value, error = minuit.values[name], minuit.errors[name]
    print(f"  Fit value: {name} = {value:.5f} +/- {error:.5f}")

gmc, gmcsigma = fit_mu, fit_sigma
print(f'Monte Carlo Method: g is {gmc} with sigma={gmcsigma}')

#____________________________ Analytical Method _______________________________
#Four variables to account for, each with their own uncertainty.
#Declare symbols and define formula
xs, ys, zs, qs = symbols('xs ys zs qs')
expr1 = ( xs / sin( ys * pi / 180 ) ) * ( 1 + ( 2/5 ) * ( zs**2 ) / ( - qs**2 + zs**2 ) )

#Find derivative and convert to numerical functions
x1diff = lambdify([xs, ys, zs, qs], expr1.diff(xs))
y1diff = lambdify([xs, ys, zs, qs], expr1.diff(ys))
z1diff = lambdify([xs, ys, zs, qs], expr1.diff(zs))
q1diff = lambdify([xs, ys, zs, qs], expr1.diff(qs))

#Evaluate at mean
x1diffeval = x1diff(x, y, z, q)
y1diffeval = y1diff(x, y, z, q)
z1diffeval = z1diff(x, y, z, q)
q1diffeval = q1diff(x, y, z, q)

#Compute g and gsigma
ganalsigma = np.sqrt( (dx**2)*(x1diffeval**2) + (dy**2)*(y1diffeval**2) + (dz**2)*(z1diffeval**2) + (dq**2)*(q1diffeval**2) )
ganal = incline(x, y, z, q)
print(f'Analytical Method: g is {ganal} with sigma={ganalsigma}')

#__________________________________ Plotting __________________________________
fig, (ax1, ax2) = plt.subplots(figsize=(13, 4), ncols=2,tight_layout=True)

s = [inversesinus(ygauss[i]) for i in range(Npoints)]

hist1 = ax1.hist(g, bins=nbins, range=(9.5, 10), histtype='step', linewidth=2, color='grey', label='Simulated Error')
hist2 = ax2.hist(s, bins=nbins, range=(4, 4.3), histtype='step', linewidth=2, color='blue', label='Inverse Sinus Error')

binwidth = (10-9.5) / nbins 
x_fit = np.linspace(10, 9.5, Npoints) 
y_fit = binwidth*pdf_extended(x_fit, fit_N, fit_mu, fit_sigma)
ya_fit = binwidth*pdf_extended(x_fit, Npoints, ganal, ganalsigma)

ax1.plot(x_fit, y_fit, '-', color='red', label='Monte Carlo Error')
ax1.plot(x_fit, ya_fit, '-', color='blue', label='Analytical Error')
ax1.set(xlabel="g [m/s^2]",
       title="Error Propagation", 
       ylabel=f"Frequency / {binwidth}")
ax1.axvline(x=ganal,color='black',label=f'g={np.round(ganal,2)}+/-{np.round(ganalsigma,2)}')
ax1.legend(loc='upper right')
ax2.set(xlabel="1 / sin ( theta )",
       title="Is it Gaussian?",)
plt.show()