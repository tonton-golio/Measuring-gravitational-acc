import streamlit as st



'# AppStat Project 1'
# brief overview

st.markdown(r'''

## Intro
The gravitational acceleration constant $g$ can be determined by fundamental concepts in classical mechanics. In this paper, we present two methods for this, using a pendulum and rolling a ball down an inclined plane. In the pendulum method, the relationship between the length of a pendulum and its period is utilized to calculate $g$. In the inclined plane method, the acceleration of the ball is used to determine $g$. The results of these methods are compared and discussed in terms of their accuracy and limitations.

For these methods to be valid; we make a series of key assumptions:
* rolling without slipping (I guess we could calculate whether we will slip.)
* etc.

## Experimental Setup and Methods
The experiments to determine the value of $g$ were carried out using the pendulum method and the inclined plane method.

### Pendulum Method

The pendulum method involved suspending a pendulum from a fixed point and allowing it to swing back and forth. The length of the pendulum was measured, and the time it took for the pendulum to complete one full swing (the period) was also measured. This was repeated for pendulums of different lengths to determine the relationship between the length of the pendulum and the period.

The pendulum used in the experiment was a simple pendulum consisting of a metal weight attached to a latex string. The length of the pendulum was measured from the point of suspension to the center of the ball. The period of the pendulum was measured using a stopwatch. The pendulum was allowed to swing for several cycles, and the period was measured each time.

### Inclined Plane Method

The inclined plane method involved rolling a ball down an inclined plane and measuring the time it took for the ball to reach the bottom. The angle of the inclined plane was measured using a protractor, and the acceleration of the ball was determined by dividing the distance the ball traveled by the time it took to travel that distance.

The inclined plane used in the experiment was a wooden board inclined at a fixed angle. A ball was placed at the top of the inclined plane, and timing gates were placed at the top and bottom to measure the time it took for the ball to travel down the incline. The angle of the inclined plane was measured using a protractor, and the acceleration of the ball was determined by dividing the distance the ball traveled by the time it took to travel that distance.

## Data Collection and Analysis

Data was collected for both the pendulum and inclined plane experiments. For the pendulum experiment, the length of the pendulum and the period of the pendulum were measured for each trial. For the inclined plane experiment, the angle of the inclined plane, the distance the ball traveled, and the time it took for the ball to travel that distance were measured for each trial.

### Error propagation
#### Pendulum
To propagate the error in L and T onto g, the error in g can be calculated using the formula for the error in a product, which is the sum of the individual errors in each variable multiplied by the derivative of g with respect to that variable. In the case of the pendulum method, where g is calculated using the formula g = 4π^2L/T^2, the error in g can be calculated as follows:

Error in g = (Error in L)(dg/dL) + (Error in T)(dg/dT)
= (Error in L)(-4π^2/T^2) + (Error in T)(-8π^2L/T^3)

This formula can be used to calculate the error in g using the individual errors in L and T. The resulting error in g can then be used to determine the accuracy of the g value calculated using the pendulum method.

#### Incline
To propagate the error in L and T onto g, the error in g can be calculated using the formula for the error in a product, which is the sum of the individual errors in each variable multiplied by the derivative of g with respect to that variable. In the case of the pendulum method, where g is calculated using the formula g = 4π^2L/T^2, the error in g can be calculated as follows:

Error in g = (Error in L)(dg/dL) + (Error in T)(dg/dT)
= (Error in L)(-4π^2/T^2) + (Error in T)(-8π^2L/T^3)

This formula can be used to calculate the error in g using the individual errors in L and T. The resulting error in g can then be used to determine the accuracy of the g value calculated using the pendulum method. Similar calculations can be performed for the inclined plane method, where g is calculated using the formula g = a/sin(θ).

''')