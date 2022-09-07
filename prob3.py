# -*- coding: utf-8 -*-
"""Prob3_MM

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MkFgAGND_AB2j-fPN2CqlNI0aV0aRBLg

#Global functions and imports
"""

import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

def measures(y, y_cap):
  n = len(y)
  sse = 0
  sst = 0
  y_bar = sum(y)/n
  for i in range(n):
    diff = y[i] - y_cap[i]
    sse += diff ** 2
    sst += (y[i] - y_bar) **2

  correlation = 1 - (sse/sst)
  return sse, sst, correlation

def linear_regression(x,y):
  n = len(x)
  xy = np.dot(np.array(x), np.array(y))

  x_sq = 0
  for i in x:
    x_sq += i**2


  a = (((n*xy) - (sum(x)*sum(y))) /((n*x_sq) - (pow(sum(x),2))))
  b = ((sum(y) -(a*sum(x)))/n)

  return a,b

def time_series(x, y):

  n = len(x)
  xy = np.dot(np.array(x),np.array(y))
  x_sq_sum = sum([pow(i,2) for i in x])
  a = (((x_sq_sum)-(n*statistics.mean(x)*statistics.mean(y)))/(x_sq_sum - (n*pow(statistics.mean(x),2))))
  b = (statistics.mean(y) - (a*statistics.mean(x)))

  return a, b

"""#Problem 1

    The following data are a result of an investigation as to the effect of reaction temperature x on 
    Percent conversion of a chemical process y. (See Myers, Montgomery and Anderson-Cook,   
    2009.) Fit a simple linear regression, and use a lack-of-fit test to determine if the model is 
    adequate. Discuss
                  
    Temperature    200  250  200  250  189.65  260.36  225  225  225  225  225  225
    Conversion     43   78   69   73   48      78      65   74   76   79   82   81
"""

x = [200, 250, 200, 250, 189.65, 260.35, 225, 225, 225, 225, 225, 225]
y = [43, 78, 69, 73, 48, 78, 65, 74, 76, 79, 83, 81]

a, b = linear_regression(x, y)
y_cap = [((a*i)+b) for i in x]
print(y_cap)

fig = plt.figure(figsize=(12,8))
plt.scatter(x, y, label = "x versus y")
plt.plot(x, y_cap, label = "x versus y_cap")
plt.legend()
sse, sst, correlation = measures(y,y_cap)
print("Correlation: ", correlation)
print("SSE: ", sse)
print("SST: ", sst)

plt.show()
t, p = stats.ttest_ind(y, y_cap)
print(f"T-test values - t: {t}, p: {p}")
print("Null hypothesis: y_cap and y are equal - model is fitting")
print("Alternate hypothese: y_cap and y are not equal - model is not fitting")
print("Threshold: 5%")
#trend
if(p > 0.95):
    print("Null hypothesis is not rejected")
else:
    print("Null hypothesis is accepted")

"""#Problem 2

    Transistor gain between emitter and collector in an integrated circuit device (hFE) is related
    to two variables (Myers, Montgomery and Anderson-Cook,2009) that can be controlled at the deposition process, 
    emitter drive-in time (x1, in minutes) and emitter dose (x2, in ions × 1014). Fourteen samples were observed 
    following deposition, and the resulting data are shown
    in the table below. We will consider linear regression models using gain as the response and 
    emitter drive-in time or emitter dose as the regressor variable

    x1 195 255  195 255  255  255  255  195 255  255  255  255  255  340
    x2 4.0 4.0  4.6 4.6  4.2  4.1  4.6  4.3 4.3  4.0  4.7  4.3  4.7  4.3
    y 1004 1636 852 1506 1272 1270 1269 903 1555 1260 1146 1276 1225 1321

    a) Determine if emitter drive-in time influences gain in a linear relationship. That is, test H0: b1 = 0, 
    where b1 is the slope of the regressor variable.
    b) Do a lack of fit test to determine if the linear relationship is adequate. Draw conclusions
    c) Determine if emitter dose influences gain in a linear relationship. Which regressor variable is the better predictor of the gain?



"""

x1 = [195, 255, 195, 255, 255, 255, 255, 195, 255, 255, 255, 255, 255, 340] #Emitter drive-in
x2 = [4.0, 4.0, 4.6, 4.6, 4.2, 4.1, 4.6, 4.3, 4.3, 4.0, 4.7, 4.3, 4.72, 4.3] #Emitter dose
y = [1004, 1636, 852, 1506, 1272, 1270, 1269, 903, 1555, 1260, 1146, 1276, 1225, 1321] #gain

print("Emitter drive-in")
a, b = linear_regression(x1, y)
print(a, b)
y_cap = [((a*i)+b) for i in x1]

plt.scatter(x1, y, label = "Original")
plt.plot(x1, y_cap, label = "Predicted")
plt.legend()
plt.show()

t, p1 = stats.ttest_ind(y, y_cap)
print(f"T-test values - t: {t}, p: {p1}")
print("Null hypothesis: y_cap and y are equal - model is fitting")
print("Alternate hypothese: y_cap and y are not equal - model is not fitting")
print("Threshold: 5%")
#trend
if(p1 > 0.05):
    print("Null hypothesis is not rejected")
else:
    print("Null hypothesis is accepted")

print("Emitter dose")
a, b = linear_regression(x2, y)
y_cap = [((a*i)+b) for i in x2]

plt.scatter(x2, y, label = "Original")
plt.plot(x2, y_cap, label = "Predicted")
plt.legend()
plt.show()

t, p2 = stats.ttest_ind(y, y_cap)
print(f"T-test values - t: {t}, p: {p2}")
print("Null hypothesis: y_cap and y are equal - model is fitting")
print("Alternate hypothese: y_cap and y are not equal - model is not fitting")
print("Threshold: 5%")
#trend
if(p2 > 0.05):
    print("Null hypothesis is not rejected")
else:
    print("Null hypothesis is rejected")

if p1 > p2:
  print("Drive-in influences gain more")
else:
  print("Dose influences gain more")

"""#Problem 3

    Evaluating nitrogen deposition from the atmosphere is a major role of the National Atmospheric
    Deposition Program (NADP), a partnership of many agencies. NADP is studying atmospheric deposition and its effect
    on agricultural crops, forest surface waters, and other resources. Nitrogen oxides 
    may affect the ozone in the atmosphere and the amount of pure nitrogen in the air we breathe. 
    The data are as follows:

    year 1978  1979  1980  1981  1982  1983  1984  1985  1986  1987  1988  1989  1990  1991  1992  1993  1994  1995  1996  1997  1998  1999
    NO   0.73  2.55  2.90  3.83  2.53  2.77  3.93  2.03  4.39  3.04  2.41  5.07  2.95  3.14  3.44  3.63  4.50  3.95  5.24  3.30  4.36  3.33

    a) Plot the data
    b) Fit a linear regression model and find R^2
    c) What can you say anout the trend in nitrogen oxide across time

"""

year = [1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]
no = np.array([0.73, 2.55, 2.9, 3.83, 2.53, 2.77, 3.93, 2.03, 4.39, 3.04, 2.41, 5.07, 2.95, 3.14, 3.44, 3.63, 4.5, 3.95, 5.24, 3.3, 4.36, 3.33])

fig = plt.figure(figsize = (12, 8))
plt.plot(year, no, label = "Year versus Nitrogen Oxide")
plt.legend()
plt.show()

a, b = linear_regression(year, no)
y_cap = np.array([((a*i)+b) for i in year])
error = no - y_cap
plt.figure(figsize = (12, 8))
plt.plot(year, no, 'o', year, no, label = "Year vs NO")
plt.plot(year, error, 'k', label = "Year vs Error")
plt.plot(year, y_cap, 'g', label = "Year vs Predicted NO")
plt.title("Linear regression Plot")
plt.legend()
plt.show()

sse, sst, correlation = measures(y,y_cap)
r_2 = correlation**2
print("R square: ", r_2)

t, p2 = stats.ttest_ind(y, y_cap)
print(f"T-test values - t: {t}, p: {p2}")
print("Null hypothesis: y_cap and y are equal - model is fitting")
print("Alternate hypothese: y_cap and y are not equal - model is not fitting")
print("Threshold: 5%")

if(p2 > 0.95):
    print("Null hypothesis is not rejected")
else:
    print("Null hypothesis is rejected")

"""#Problem 4

    In the following data, V represents a mean walking velocity and P represents the population size.
    We wish to know if we can predict the population size P by observing how fast people walk.
    a) Plot the data.
    b) What kind of a relationship is suggested?
    c) Test the following models by plotting the appropriate transformed data.
      i)  P = a V^b --> log P = log a + b*log V
      ii) P = a ln V

    V 2.27 2.76 3.27  3.31 3.70  3.85  4.31  4.39   4.42   4.81   4.90 
    P 2500 365  23700 5491 14000 78200 70700 138000 304500 341948 49375 

    V 5.05   5.21   5.62    5.88
    P 260200 867023 1340000 1092759
"""

v = [2.27, 2.76, 3.27, 3.31, 3.7, 3.85, 4.31, 4.39, 4.42, 4.81, 4.9, 5.05, 5.21, 5.62, 5.88]
p = [2500, 365, 23700, 5491, 14000, 78200, 70700, 138000, 304500, 341948, 49375, 260200, 867023, 1340000, 1092759]

l_v = [math.log(i) for i in v]
l_p = [math.log(i) for i in p]
plt.figure(figsize = (12, 8))
plt.plot(v, p, 'o', v, p, label = "P vs V")
plt.show()
print()
plt.figure(figsize = (12, 8))
plt.plot(l_v, l_p, 'o', l_v, l_p, label = "Ln P vs Ln V")
plt.legend()
plt.show()    #Exponential

b, a = linear_regression(l_v, l_p)
y_cap = [math.exp(b*i + a) for i in l_v]
plt.figure(figsize = (12, 8))
plt.scatter(v, p, label = "Original Data")
plt.plot(v, y_cap, 'orange', label = "Regression curve")
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(l_v, p, label = "Ln v vs P")
plt.legend()
plt.show()

a, b = linear_regression(l_v, p)
y_cap = []
for i in l_v:
  y_cap.append(a*i + b)

plt.figure(figsize = (12, 8))
plt.scatter(v, p, color='orange', label = "Original")
plt.plot(v, y_cap, label = "Predicted")
plt.legend()
plt.show()

"""#Problem 5

    In 1601, the German astronomer Johannes Kepler became director of the Prague Observatory. Kepler had been helping 
    Tycho Brahe in collecting 13 years of observations on the relative motion of the planet Mars.
    i.  Each planet moves in an ellipse with the sun at one focus
    ii. Each planet, the line from the sun to the planet sweeps out equal areas in each times

    a) Plot the period time T versus the mean distance r

    Planet: Me   Ve    Ea    Ma    Ju    Sa     Ur     Ne
    Period: 88   225   365   687   4329  10753  30660  60150
    Mean r: 57.9 108.2 149.6 227.9 778.1 1428.2 2837.9 4488.9

    b) Assuming, T = Cr^a, determine C and a, by plotting T vs ln r. Is it reasonable? Formulate Kepler's third law

"""

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
periods = [88, 225, 365, 687, 4329, 10753, 30660, 60150]
r = [57.9, 108.2, 149.6, 227.9, 778.1, 1428.2, 2837.9, 4488.9]

fig = plt.figure(figsize = (12, 8))
plt.plot(r, periods, label = "T vs r")
plt.yticks(ticks = periods, labels = planets)
plt.scatter(r, periods)
plt.title("Complete List")
plt.xlabel("Radius")
plt.ylabel("Periods")
plt.legend()
plt.show()
print()
fig = plt.figure(figsize = (12, 8))
plt.plot(r[:4], periods[:4], label = "T vs r")
plt.yticks(ticks = periods[:4], labels = planets[:4])
plt.scatter(r[:4], periods[:4])
plt.title("First 4 planets")
plt.xlabel("Radius")
plt.ylabel("Periods")
plt.legend()
plt.show()

ln_r = [np.log(i) for i in r]
fig = plt.figure(figsize = (12, 8))
plt.plot(ln_r, periods, label = "T vs Ln r")
plt.yticks(ticks = periods, labels = planets)
plt.scatter(ln_r, periods)
plt.title("Complete List")
plt.xlabel("Radius")
plt.ylabel("Periods")
plt.legend()
plt.show()
print()
fig = plt.figure(figsize = (12, 8))
plt.plot(ln_r[:4], periods[:4], label = "T vs Ln r")
plt.yticks(ticks = periods[:4], labels = planets[:4])
plt.scatter(ln_r[:4], periods[:4])
plt.title("First 4 planets")
plt.xlabel("Radius")
plt.ylabel("Periods")
plt.legend()
plt.show()

l_p = [math.log(i) for i in periods]
l_r = [math.log(i) for i in r]
b, a = linear_regression(l_r, l_p)
y_cap = [math.exp(b*i + a) for i in l_p]
plt.figure(figsize = (12, 8))
plt.scatter(r, periods, label = "Original Data")
plt.plot(r, y_cap, 'orange', label = "Regression curve")
plt.legend()
plt.show()

a, b = linear_regression(l_r, l_p)
y_cap = []
for i in l_r:
  y_cap.append(math.exp(i*a + b))

plt.figure(figsize = (12, 8))
plt.plot(r, periods, 'o', r, periods, label = "Original", linewidth = 5)
plt.plot(r, y_cap, 'o', r, y_cap, label = "Predicted")
plt.yticks(ticks = periods, labels = planets)
plt.title("Complete")
plt.legend()
plt.show()

print()

plt.figure(figsize = (12, 8))
plt.plot(r[:4], periods[:4], 'o', r[:4], periods[:4], label = "Original", linewidth = 5)
plt.plot(r[:4], y_cap[:4], 'o', r[:4], y_cap[:4], label = "Predicted")
plt.yticks(ticks = periods[:4], labels = planets[:4])
plt.title("First 4 planets")
plt.legend()
plt.show()