#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm

data = np.loadtxt('salsify-user-study-webcam.csv', delimiter=',')

y = data[:,4]

dmin, dmax = min(data[:,1]), max(data[:,1])
data[:,1] = (data[:,1] - dmin) / (dmax - dmin)

dmin, dmax = min(data[:,3]), max(data[:,3])
data[:,3] = (data[:,3] - dmin) / (dmax - dmin)

#x = data[:,0]
#x = data[:,2]
x = data[:,1:4:2]
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
print('exact', exact_count, exact_count/len(y))
print('off_by_one', one_off_count, one_off_count/len(y))
print('less_than_equal_one', exact_count+one_off_count, (exact_count+one_off_count)/len(y))

