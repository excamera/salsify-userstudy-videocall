#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm

data = np.loadtxt('salsify-user-study-webcam.csv', delimiter=',')

y = data[:,3]

dmin, dmax = min(data[:,0]), max(data[:,0])
data[:,0] = (data[:,0] - dmin) / (dmax - dmin)

dmin, dmax = min(data[:,2]), max(data[:,2])
data[:,2] = (data[:,2] - dmin) / (dmax - dmin)

#x = data[:,0]
#x = data[:,2]
x = data[:,0:3:2]
x = sm.add_constant(x)

results = sm.OLS(endog=y, exog=x).fit()

print(results.summary())

exact_count = 0
one_off_count = 0
fun = lambda x: 3.8642 + -2.5895*x[0] + 0.4891*x[1] 
for i in range(len(y)):
    xx = x[i,:]
    yy = y[i]
    
    res = abs(yy - fun(xx[1:]))
    if res < 0.5:
        exact_count += 1
    elif res < 1.5:
        one_off_count += 1

print('total', len(y))
print('exact', exact_count)
print('off_by_one', one_off_count)
    
