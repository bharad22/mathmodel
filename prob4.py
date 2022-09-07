# -*- coding: utf-8 -*-
"""Prob4_MM

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tIh6Lk8G-nX_4DRbzjDqz9T-m1_3lVil

#Global functions and imports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def  linear_model(x, y):
    n = len(x)
    xy = 0
    x2 = 0
    for i in range(n):
        x2 += x[i] * x[i]
        xy += x[i] * y[i]
    m = (n * xy - sum(x) * sum(y)) / (n * x2 - sum(x) * sum(x)) 
    c = (sum(y) - m * (sum(x))) / n
    
    y_cap = []
    for i in x:
        y_cap.append(m*i + c)
      
    plt.scatter(x, y, label = "Original data")
    plt.plot(x, y_cap, 'orange', label = "Predicted")
    plt.legend()
    plt.show()

    r2 = r2_score(y, y_cap)
    print('r2 score ', r2)

"""#Airtel data - 532454.csv - Monthly Data from July 2002 - for 20 years"""

df = pd.read_csv('532454.csv')

x = df['Open Price'].tolist()
y = df['Close Price'].tolist()

print("x = Open Price\ny = Close Price")
linear_model(x, y)
print()

x = df['High Price'].tolist()
y = df['Low Price'].tolist()

print("x = High Price\ny = Low Price")
linear_model(x, y)
print()

print("Open Price, High Price, Low Price vs Close Price")

x = df[['Open Price', 'High Price', 'Low Price']]
y = df['Close Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=100)  
model = LinearRegression()
model.fit(x_train, y_train) 

y_cap_model = model.predict(x_test)  
x_cap_model = model.predict(x_train)

print(model.predict([[20000.00, 10000.60, 30000.20]]))
print('R squared value of the model: {:.2f}'.format(model.score(x, y)))
print()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x = df['Open Price'].tolist()
y = df['High Price'].tolist()
z = df['Low Price'].tolist()
e = df['Close Price'].tolist()

plt.figure(figsize = (16, 8))
plt.plot(x, label = "Opening Price")
plt.plot(y, label = "High Price")
plt.plot(z, label = "Low Price")
plt.plot(e, label = "Closing Price")
plt.title("Time series of Airtel Data")
plt.legend()
plt.show()

fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(111, projection='3d')
print("Weights:\nOpen Price: %f, High Price: %f, Low Price: %f" %(model.coef_[0], model.coef_[1], model.coef_[2]))
img = ax.scatter(x, y, z, c = e, cmap = plt.hot())
ax.set_xlabel("Open Price")
ax.set_ylabel("High Price")
ax.set_zlabel("Low Price")
fig.colorbar(img)
plt.show()

a, b, c = model.coef_
d = model.intercept_
y_cap = []
for i, j, k in zip(x, y, z):
  y_cap.append(a*i + b*j + c*k + d)

plt.figure(figsize = (16, 8))

plt.plot(e, label = "Original")
plt.plot(y_cap, label = "Predicted")
plt.title("Original and Predicted Closing Prices in a Time series")
plt.xlabel("Time")
plt.ylabel("Closing Prices")
plt.legend()
plt.show()

t, p = stats.ttest_ind(e, y_cap)
print(p)
if p > 0.95:
  print("Model is fitting") 
else:
  print("Model is not fitting")