import streamlit as st

'# AppStat Project 1'


'''
In this experiment, we want to measure acceleration due to gravity.

We apply two methods in pursuit of this goal:
* obeservation of the period of a **pendulum**
* acceleration of a **ball on an incline** 


For these methods to be valid; we make a series of key assumptions:
* rolling without slipping (I guess we could calculate whether we will slip.)
* etc.


The main focus of this project, will be caution to uncertainties and error propagation. Hopefully, if we aren't too sloppy, the two methods should yield similar results for $g$.
'''

'''
## Text from chatGPT
### How to measure the acceleration due to gravity?
There are several ways to measure the acceleration due to gravity. One common method is to use a device called a accelerometer, which is a device that measures acceleration. Another method is to use a pendulum. To measure the acceleration due to gravity using a pendulum, you can measure the time it takes for the pendulum to swing back and forth, and then use the formula g = 4π^2 * L / T^2 to calculate the acceleration due to gravity, where L is the length of the pendulum and T is the time it takes to swing back and forth. Another method is to use a device called a gravimeter, which is a specialized instrument that measures the strength of the gravitational field.


### How to do this precisely with the pendulum
To measure the acceleration due to gravity using a pendulum, you need to make sure that you are using a high-precision pendulum and that you are performing the measurements carefully. Here are the steps to measure the acceleration due to gravity using a pendulum:

Set up the pendulum by attaching it to a fixed support and allowing it to hang freely. Make sure that the pendulum is set up in a location where it can swing freely without any interference from other objects or air currents.

Measure the length of the pendulum from the point where it is attached to the support to the bottom of the pendulum bob (the weight at the bottom of the pendulum).

Start the pendulum swinging back and forth, and use a stopwatch to measure the time it takes for the pendulum to complete a certain number of swings. For the most accurate results, you should measure the time for at least 10 swings.

Use the formula g = 4π^2 * L / T^2 to calculate the acceleration due to gravity, where L is the length of the pendulum and T is the time it takes to swing back and forth.

Repeat the measurements several times to ensure that your results are accurate and consistent.

It's important to note that the accuracy of your measurement will depend on the precision of your equipment and the care with which you perform the measurements. To achieve the most accurate results, you should use a high-quality pendulum and carefully control the conditions under which the measurements are taken.


### How do i propagate the uncertainties for this?


To propagate the uncertainties in your measurement of the acceleration due to gravity using a pendulum, you need to take into account the uncertainties in the length of the pendulum and the time it takes to swing back and forth. To do this, you can use the formula for the propagation of uncertainty, which states that the uncertainty in the result (Δg) is the square root of the sum of the squares of the individual uncertainties (ΔL and ΔT) divided by the quantity being measured (g). In this case, the formula would be:

Δg = sqrt((ΔL / L)^2 + (ΔT / T)^2) * g

To use this formula, you need to determine the uncertainties in the length of the pendulum and the time it takes to swing back and forth. The uncertainty in the length of the pendulum can be determined by measuring the length of the pendulum several times and calculating the standard deviation of the measurements. The uncertainty in the time it takes to swing back and forth can be determined by measuring the time for several swings and calculating the standard deviation of the measurements. Once you have determined the uncertainties in the length and time, you can plug them into the formula above to calculate the uncertainty in the acceleration due to gravity.

### please provide python code for this

Here is a sample Python code that demonstrates how to propagate the uncertainties in the measurement of the acceleration due to gravity using a pendulum:

```python
# Import the math module to access the square root function
import math

# Define the variables
L = 1.0 # Length of the pendulum in meters
T = 1.0 # Time for the pendulum to swing back and forth in seconds

# Define the uncertainties in the length and time
delta_L = 0.1 # Uncertainty in the length in meters
delta_T = 0.1 # Uncertainty in the time in seconds

# Calculate the acceleration due to gravity
g = 4 * math.pi**2 * L / T**2

# Calculate the uncertainty in the acceleration due to gravity
delta_g = math.sqrt((delta_L / L)**2 + (delta_T / T)**2) * g

# Print the results
print("Acceleration due to gravity = %.3f" % g)
print("Uncertainty in acceleration = %.3f" % delta_g)
```

This code first imports the math module, which provides access to the square root function. It then defines the variables L and T, which represent the length of the pendulum and the time it takes to swing back and forth, respectively. The code also defines the uncertainties in the length and time, which are assumed to be 0.1 in this example. Next, the code uses the formula for the acceleration due to gravity to calculate the value of g, and it uses the formula for the propagation of uncertainty to calculate the uncertainty in g. Finally, the code prints the results to the screen.


### Write a scientific report on measuring the acceleration due to gravity with a pendulum as well as with a ball rolling down an incline. Go into detail about the error propagation.

Measuring the acceleration due to gravity is an important task in physics and engineering, and there are several methods for doing so. In this report, we will describe two common methods for measuring the acceleration due to gravity: using a pendulum and using a ball rolling down an incline. We will also discuss the error propagation for each of these methods.

Using a Pendulum

One common method for measuring the acceleration due to gravity is to use a pendulum. To measure the acceleration due to gravity using a pendulum, we need to set up the pendulum by attaching it to a fixed support and allowing it to hang freely. We then measure the length of the pendulum from the point where it is attached to the support to the bottom of the pendulum bob (the weight at the bottom of the pendulum).

Next, we start the pendulum swinging back and forth, and use a stopwatch to measure the time it takes for the pendulum to complete a certain number of swings. For the most accurate results, we should measure the time for at least 10 swings. Finally, we use the formula g = 4π^2 * L / T^2 to calculate the acceleration due to gravity, where L is the length of the pendulum and T is the time it takes to swing back and forth.

To propagate the uncertainties in our measurement of the acceleration due to gravity using a pendulum, we need to take into account the uncertainties in the length of the pendulum and the time it takes to swing back and forth. To do this, we can use the formula for the propagation of uncertainty, which states that the uncertainty in the result (Δg) is the square root of the sum of the squares of the individual uncertainties (ΔL and ΔT) divided by the quantity being measured (g). In this case, the formula would be:

Δg = sqrt((ΔL / L)^2 + (ΔT / T)^2) * g

To use this formula, we need to determine the uncertainties in the length of the pendulum and the time it takes to swing back and forth. The uncertainty in the length of the pendulum can be determined by measuring the length of the pendulum several times and calculating the standard deviation of the measurements. The uncertainty in the time it takes to swing back and forth can be determined by measuring the time for several swings and calculating the standard deviation of the measurements. Once we have determined the uncertainties in the length and time, we can plug them into the formula above to calculate the uncertainty in the acceleration due to gravity.

Using a Ball Rolling Down an Incline

Another common method for measuring the acceleration due to gravity is to use a ball rolling down an incline. To measure the acceleration due to gravity using this method, we need to set up the incline by attaching one end of a board to a fixed support and raising the other end to a certain angle with respect to the horizontal. We then place a ball at the top of the incline and release it, allowing it to roll down the incline.

Next, we use a stopwatch to measure the time it takes for the ball to roll a certain distance down the incline. For the most accurate results, we should measure the time for several trials and calculate the average. Finally, we use the formula g = d / t^2 to calculate the acceleration due to gravity, where d is the distance the ball rolls down the incline and t is the time it takes to roll that distance.

To propagate the uncertainties in our measurement of the acceleration due to gravity using a ball rolling down an incline, we need to take into account the uncertainties in the

'''