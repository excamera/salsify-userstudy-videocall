#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_delay(xx, yy, name):
    plt.clf()
    xxx = sm.add_constant(xx)

    results = sm.OLS(endog=yy, exog=xxx).fit()

    mean, std = [],[]
    for i in [1,15,30,60]:
        select_y = []
        for j in range(len(xx)):
            if xx[j] == i:
                select_y.append(yy[j])

        mean.append( np.mean(select_y) )
        std.append( np.std(select_y) )

    print(mean,std)
    
    plt.errorbar([1,15,30,60], mean, yerr=list(map(lambda x :x, std)),
                 fmt='x', color='k', ecolor='k', capthick=2)
    plt.title('Delay v. QoE Score')
    plt.ylabel('QoE score')
    plt.xlabel('Delay (# frames)')
    plt.axis([-5, 65, 1, 5])
    plt.text(34.85, 4.80, 'mean and +/- stddev plotted')

    x = [1,60]
    y = [results.params[0] + results.params[1], results.params[0] + 60*results.params[1]]
    plt.axis([-5, 65, 1, 5])
    plt.plot(x,y, 'r--')
    
    #plt.show()
    plt.savefig(name)
    #print(results.params)

def plot_quality(xx, yy, name):
    plt.clf()
    xxx = sm.add_constant(xx)

    results = sm.OLS(endog=yy, exog=xxx).fit()

    mean, std = [],[]
    for i in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
        select_y = []
        for j in range(len(xx)):
            if abs(xx[j] - i) < 0.5:
                select_y.append(yy[j])

        mean.append( np.mean(select_y) )
        std.append( np.std(select_y) )

    plt.errorbar([10, 11, 12, 13, 14, 15, 16, 17, 18], mean, yerr=list(map(lambda x :x, std)),
                 fmt='x', color='k', ecolor='k', capthick=2)
    plt.title('Visual Quality v. QoE Score')
    plt.ylabel('QoE score')
    plt.xlabel('Visual Quality (SSIM)')
    plt.axis([9, 19, 1, 5])
    plt.text(14.8, 4.74, 'mean and +/- stddev plotted')

    x = [10,18]
    y = [results.params[0] + 10*results.params[1], results.params[0] + 18*results.params[1]]
    plt.plot(x,y, 'r--')

    #plt.scatter(xx,yy)
    
    #plt.show()
    plt.savefig(name)
    #print(results.params)

data = np.loadtxt('salsify-user-study-webcam.csv', delimiter=',')

# dmin, dmax = min(data[:,1]), max(data[:,1])
# data[:,1] = (data[:,1] - dmin) / (dmax - dmin)

# dmin, dmax = min(data[:,3]), max(data[:,3])
# data[:,3] = (data[:,3] - dmin) / (dmax - dmin)

y = data[:,4]

x = data[:,1]
plot_delay(x, y, 'delay.svg')

x = data[:,3]
plot_quality(x, y, 'quality.svg')

#x = data[:,1:4:2]
#plot(x, y, '2d_linear.svg')
