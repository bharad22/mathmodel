import matplotlib.pyplot as plt
from math import factorial
import numpy as np
import math
from scipy import stats, interpolate

def ForwardInterpolationTable(arr):
  table =[]
  while len(arr) > 1:
    rowInst = [arr[i] - arr[i-1] for i in range(1, len(arr))]
    arr = rowInst
    table.append(rowInst)

  return table

def comb(n, r, forward = True):
  val = 1
  for i in range(r):
    val *= ((n-i) if forward else (n+i))
  return val if r == 1 else val / math.factorial(r)

def ForwardInterpolationPolynomial(initY, fInrTable, u):
  re = initY
  for i in range(len(fInrTable)): 
    re += comb(u, i+1) * fInrTable[i][0]
  return re

def BackwardInterpolationPolynomial(initY, fInrTable, u):
  re = initY
  for i in range(len(fInrTable)):
    re += comb(u, i+1, False) * fInrTable[i][-1]
  return re

def LagrangeInterpolation(x,y,x_val):
  re = 0
  for i in range(len(x)):
    numerator=1
    denominator = 1
    for j in range(len(x)):
      if j!=i:
        numerator *= (x_val - x[j])
        denominator *= (x[i] - x[j])
    re += ((numerator/denominator)*y[i])

  return re

def NewtonDividedDifference(x,y,n):
  divided_difference = [[0 for i in range(10)] 
        for j in range(10)]

  for i in range(n):
    divided_difference[i][0]=y[i]
    
  for i in range(1, n): 
    for j in range(n - i): 
        divided_difference[j][i] = ((divided_difference[j][i - 1] - divided_difference[j + 1][i - 1]) /(x[j] - x[i + j]));
  return divided_difference

def NewtonFormula(value, x, y, n):
  def previous_term(i, value, x): 
    prev = 1; 
    for j in range(i): 
        prev = prev * (value - x[j]); 
    return prev
  
  sum_temp = y[0][0]; 

  for i in range(1, n):
      sum_temp = sum_temp + (previous_term(i, value, x) * y[0][i]); 
    
  return sum_temp

def LinearRegression(x, y):
  n = len(x)
  xy = np.dot(np.array(x), np.array(y))

  x_sq = 0
  for i in x:
    x_sq += i**2


  a = (((n*xy) - (sum(x)*sum(y))) /((n*x_sq) - (pow(sum(x),2))))
  b = ((sum(y) -(a*sum(x)))/n)

  return a,b

def CalculateMeasures(y, y_cap):
  n = len(y)
  sse = 0
  sst = 0
  y_bar = sum(y)/n
  for i in range(n):
    diff = y[i] - y_cap[i]
    sse += (y[i] - y_cap[i]) * (y[i] - y_cap[i])
    sst += (y[i] - y_bar) * (y[i] - y_bar)

  correlation = 1-(sse/sst)
  return sse, sst, correlation

def predict_val(b0, b1, x):
  return [b0 + b1 * i for i in x]