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

def ComputeSeasonalIndex(x, num_yrs):
  
  four_quarter_moving_avg = []
  four_quarter_centered_moving_avg = []
  percent_actual_to_moving_avg = []
  j = 0

  for i in range(len(x)-3):
    temp = (x[i] + x[i+1] + x[i+2] + x[i+3]) / 4
    four_quarter_moving_avg.append(temp)

  print("\nFour quarter moving averages: ", four_quarter_moving_avg)

  for i in range(len(four_quarter_moving_avg)-1):
    temp = (four_quarter_moving_avg[i] + four_quarter_moving_avg[i+1]) / 2
    four_quarter_centered_moving_avg.append(temp)

  print("\nFour quarter centered moving averages: ", four_quarter_centered_moving_avg)

  for i in range(2, len(x)-2):
    temp = (x[i] / four_quarter_centered_moving_avg[j]) * 100
    j += 1
    percent_actual_to_moving_avg.append(temp)

  print("\nPercentage of actual to moving averages: ", percent_actual_to_moving_avg)

  diff = [0, 0] + percent_actual_to_moving_avg
  n = len(diff) % 4
  diff += [0 for i in range(n)]

  print("\nDiff values: ", diff)

  track = []
  modified_mean = []
  width = len(diff) // num_yrs

  for i in range(4):
    temp = []

    for j in range(0, num_yrs):
      temp.append(diff[width*j+i])

    track.append(temp)
  
  print("\nTrack values: ", track)

  for i in range(len(track)):
    track[i] = [i for i in track[i] if i != 0]
    a = min(track[i])
    b = max(track[i])
    track[i].remove(a)
    track[i].remove(b)
    n = len(track[i])
    modified_mean.append(sum(track[i]) / n)

  print("\nModified means / Trimmed means: ", modified_mean)

  tot = sum(modified_mean)
  adjusting_factor = 400 / tot
  seasonal_indices = []

  for i in range(len(modified_mean)):
    seasonal_indices.append(modified_mean[i] * adjusting_factor)

  print("\nSeasonal indices: ", seasonal_indices)

  modified_seasonal_indices = [i/100 for i in seasonal_indices]
  modified_seasonal_indices = modified_seasonal_indices * num_yrs

  deseasonalized_data = []

  for i in range(len(x)):
    deseasonalized_data.append((x[i] / modified_seasonal_indices[i]))

  print("\nDeseasonalized data: ", deseasonalized_data)

  ans_dict = {'four_quarter_moving_avg':four_quarter_moving_avg, 'four_quarter_centered_moving_avg':four_quarter_centered_moving_avg, 'percent_actual_to_moving_avg':percent_actual_to_moving_avg, 'diff':diff, 'track':track, 'modified_mean':modified_mean, 'seasonal_indices':seasonal_indices, 'deseasonalized_data':deseasonalized_data}

  return ans_dict

def IdentifyTrend(x, num_yrs):
  n = len(x)
  coding = [0 for i in range(n)]

  mid = (n//2) - 1
  coding[mid] = -0.5
  coding[mid+1] = 0.5

  for i in range(mid-1, -1, -1):
    coding[i] = coding[i+1] - 1

  for i in range(mid+2, n):
    coding[i] = coding[i-1] + 1

  for i in range(n):
    coding[i] *= 2

  xy = [i*j for i,j in zip(coding, x)]
  x_2 = [i**2 for i in coding]

  sum_y = sum(x)
  sum_x_2 = sum(x_2)
  sum_xy = sum(xy)

  print("\nSummation y: ", sum_y)
  print("\nSummation xy: ", sum_xy)
  print("\nSummation x2: ", sum_x_2)

  b = sum_xy / sum_x_2
  a = sum_y / (num_yrs * 4)

  ans_dict = {'a':a, 'b':b, 'coding':coding}
  
  return ans_dict

def Arma(p, q, dataColumn):
  # AR
  reg = LinearRegression()
  n = len(dataColumn)
  Y = np.array([dataColumn[i: n-p+i] for i in range(p+1)])[::-1]
  y, x = Y[0].transpose(), Y[1:].transpose()

  reg.fit(x,y)
  y_pred=reg.predict(x)
  residual = np.subtract(y, y_pred)
  print("AR equation", np.array([reg.coef_[0], reg.coef_[1], reg.intercept_]))
  
  # MA
  reg = LinearRegression()
  n = len(residual)
  X = np.array([residual[i: n-q+i] for i in range(q+1)])[::-1]
  y, x = X[0].transpose(), X[1:].transpose()

  reg.fit(x,y)
  print("MA equation", np.array([reg.coef_[0], reg.coef_[1], reg.intercept_]))