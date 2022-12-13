#Error propagation for a ball on an incline experiment
import sys
#sys.path.append('C:/Users/micha/OneDrive - University of Copenhagen/Kode/Applied Statistics/GitHub/External_Functions')
sys.path.append('/home/asp/Downloads/Applied Statistics/AppStat2022-main/External_Functions')
import numpy as np
import sympy as sp
from iminuit import Minuit    
import matplotlib.pyplot as plt
from ExternalFunctions import UnbinnedLH, BinnedLH, Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax
from math import pi

def deg_to_rad(alpha):
    return alpha*pi/180

def incline(a, theta, D_ball, L_rail, del_theta=0):
    A = 1+2/5*(D_ball**2)/(D_ball**2 - L_rail**2)
    B = a/np.sin(deg_to_rad(theta+del_theta))
    return A*B

def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(x-mu)**2/sigma**2)

def gauss_norm(x, mu, sigma, N) :
    return N * gauss(x, mu, sigma)

#Setting values
#a:x, theta:y, ball:z, rail:q
#Acceleration fit and uncertainty
a, a_err = 1.5743, 0.0367              #[m/s^2]
#Angle of table and uncertainty
#Setup 000deg
theta, theta_err = 14.00, 0.08         #[Deg]
#Setup 180deg
#y, dy = 13.70, 0.11        #[Deg]
#Big ball
D_ball_L, D_ball_L_err = 15.015, 0.003       #[mm]
#Small ball
#D_ball_s, D_ball_s_err = 12.728, 0.004      #[mm]
#Length of rail and uncertainty
L_rail, L_rail_err = 5.950, 0.006        #[mm]


#____________________________ Monte Carlo Method ______________________________
r = np.random
r.seed(42)
Npoints=100000

#Simulating data
a_gauss = r.normal(loc=a, scale=a_err, size=Npoints)
theta_gauss = r.normal(loc=theta, scale=theta_err, size=Npoints)
D_ball_L_gauss = r.normal(loc=D_ball_L, scale=D_ball_L_err, size=Npoints)
L_rail_gauss = r.normal(loc=L_rail, scale=L_rail_err, size=Npoints)

nbins = 100

#Evaluating function
g = [incline(a_gauss[i], theta_gauss[i], D_ball_L_gauss[i], L_rail_gauss[i]) for i in range(Npoints)]

#Fitting gaussian with Minuit
binned_likelihood = BinnedLH(gauss_norm, 
                             g,
                             bins=nbins, 
                             extended=True)
Minuit.print_level = 1
minuit = Minuit(binned_likelihood, mu=9.81, sigma=0.1, N=Npoints) 
minuit.errordef = Minuit.LIKELIHOOD
minuit.migrad() #Initiate minuit with migrad

fit_mu, fit_sigma, fit_N = minuit.values[:]     
for name in minuit.parameters:
    value, error = minuit.values[name], minuit.errors[name]
    print(f"  Fit value: {name} = {value:.5f} +/- {error:.5f}")

gmc, gmcsigma = fit_mu, fit_sigma
print(f'Monte Carlo Method: g is {gmc} with sigma={gmcsigma}')

#____________________________ Analytical Method _______________________________
#Four variables to account for, each with their own uncertainty.
#Declare symbols and define formula
xs, ys, zs, qs = sp.symbols('xs ys zs qs')
expr1 = ( xs / sp.sin( ys * pi / 180 ) ) * ( 1 + ( 2/5 ) * ( zs**2 ) / ( - qs**2 + zs**2))

#Find derivative and convert to numerical functions
x1diff = sp.lambdify([xs, ys, zs, qs], expr1.diff(xs))
y1diff = sp.lambdify([xs, ys, zs, qs], expr1.diff(ys))
z1diff = sp.lambdify([xs, ys, zs, qs], expr1.diff(zs))
q1diff = sp.lambdify([xs, ys, zs, qs], expr1.diff(qs))

#Evaluate at mean
x1diffeval = x1diff(a, theta, D_ball_L, L_rail)
y1diffeval = y1diff(a, theta, D_ball_L, L_rail)
z1diffeval = z1diff(a, theta, D_ball_L, L_rail)
q1diffeval = q1diff(a, theta, D_ball_L, L_rail)

#Compute g and gsigma
ganalsigma = np.sqrt( (a_err**2)*(x1diffeval**2) + (theta_err**2)*(y1diffeval**2) + (D_ball_L_err**2)*(z1diffeval**2) + (L_rail_err**2)*(q1diffeval**2) )
ganal = incline(a, theta, D_ball_L, L_rail)
print(f'Analytical Method: g is {ganal} with sigma={ganalsigma}')

#__________________________________ Plotting __________________________________
fig, (ax1, ax2) = plt.subplots(figsize=(13, 4), ncols=2,tight_layout=True)

arctheta = [np.arcsin(deg_to_rad(theta_gauss[i])) for i in range(Npoints)]

hist1 = ax1.hist(g, bins=nbins, range=(9.81-5, 9.81+5), histtype='step', linewidth=2, color='grey', label='Simulated Error')
hist2 = ax2.hist(arctheta, bins=nbins, range=(4, 4.3), histtype='step', linewidth=2, color='blue', label='Inverse Sinus Error')

binwidth = (10-9.5) / nbins 
x_fit = np.linspace(10, 9.5, Npoints) 
y_fit = binwidth*gauss_norm(x_fit, fit_mu, fit_sigma, fit_N)
ya_fit = binwidth*gauss_norm(x_fit, ganal, ganalsigma, Npoints)

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