#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import seaborn as sns
# import pandas as pd

# sns.set(color_codes=True)
# sns.set_style("white")

def plot_delay(xx, yy, name):
    plt.clf()
    xxx = sm.add_constant(xx)

    results = sm.OLS(endog=yy, exog=xxx).fit()

    mean, std = [],[]
    r = [1,15,30,60]
    for i in [1,15,30,60]:
        select_y = []
        for j in range(len(xx)):
            if xx[j] == 33*i:
                select_y.append(yy[j])

        mean.append( np.mean(select_y) )
        std.append( np.std(select_y) )

    print(mean,std)
    # x = np.asarray(r)
    # y = np.asarray(mean)
    # u = np.asarray(std)
    # print(x, y, u)
    # exercise = sns.load_dataset("exercise")
    
    # ax = sns.tsplot(data=y, time=x, unit=u, err_style="ci_bars", interpolate=False)

    plt.axis([-100, 2100, 0.75, 5.25])
    plt.title('Delay v. QoE Score')
    plt.ylabel('QoE score')
    plt.xlabel('Delay (milliseconds)')
    
    p = plt.errorbar(list(map(lambda x: 33*x,[1,15,30,60])), mean, yerr=list(map(lambda x :x, std)),
                     fmt='o', color='k', ecolor='k', capsize=2, capthick=2, lw=2,
                     label='mean ± std')

    x = [0, 2050]
    y = [results.params[0] + x[0]*results.params[1], results.params[0] + x[1]*results.params[1]]
    
    #plt.text(34.85, 4.80, 'mean and +/- stddev plotted')

    pp = plt.plot(x,y,'r--',label='regression line')

    #plt.legend()
    patch = mpatches.Patch(color='white', label='R² = ' + str(round(results.rsquared,2)))
    #plt.legend([p,pp[0]], ['mean ± std', 'regression line'])
    plt.legend(handles=[p, pp[0], patch], labels=['mean ± std', '-x/'+str(round(-1/results.params[1],2))+' + ' + str(round(results.params[0],2)), 'R² = ' + str(round(results.rsquared,3))])
    #plt.legend(handles=[p, pp[0], patch], labels=['mean ± std', 'regression line', 'R² = ' + str(round(results.rsquared,3))])
    #    plt.legend([p,pp[0]], ['mean ± std', 'regression line'])
    
    #plt.show()
    plt.savefig(name)

def plot_quality(xx, yy, name):
    plt.clf()
    xxx = sm.add_constant(xx)

    results = sm.OLS(endog=yy, exog=xxx).fit()

    mean, std = [],[]
    for i in [10, 12, 14, 16, 18]:
        select_y = []
        for j in range(len(xx)):
            if abs(xx[j] - i) <= 1.0:
                select_y.append(yy[j])

        mean.append( np.mean(select_y) )
        std.append( np.std(select_y) )

    p = plt.errorbar([10, 12, 14, 16, 18], mean, yerr=list(map(lambda x :x, std)),
                 fmt='o', color='k', ecolor='k', capsize=2, capthick=2, lw=2,
                 label='mean ± std')
    
    plt.title('Visual Quality v. QoE Score')
    plt.ylabel('QoE score')
    plt.xlabel('Visual Quality (dB SSIM)')
    plt.axis([9, 19, 0.75, 5.25])

    x = [9.5,18.5]
    y = [results.params[0] + x[0]*results.params[1], results.params[0] + x[1]*results.params[1]]
    pp = plt.plot(x,y, 'r--', label='regression line')

    patch = mpatches.Patch(color='white', label='R² = ' + str(round(results.rsquared,2)))
    #plt.legend([p,pp[0]], ['mean ± std', 'regression line'])
    plt.legend(handles=[p, pp[0], patch], labels=['mean ± std', str(round(results.params[1],2))+'x + ' + str(round(results.params[0],2)), 'R² = ' + str(round(results.rsquared,3))])
    #plt.legend(handles=[p, pp[0], patch], labels=['mean ± std', 'regression line', 'R² = ' + str(round(results.rsquared,3))])
    #plt.legend([p,pp[0]], ['mean ± std', 'regression line'])

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

print('plotting delay')
x = 33*data[:,1]
plot_delay(x, y, 'delay.png')
plot_delay(x, y, 'delay.svg')


print('plotting quality')
x = data[:,3]
plot_quality(x, y, 'quality.png')
plot_quality(x, y, 'quality.svg')

#x = data[:,1:4:2]
#plot(x, y, '2d_linear.svg')
