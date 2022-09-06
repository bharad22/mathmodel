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
