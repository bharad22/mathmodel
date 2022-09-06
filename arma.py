import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

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