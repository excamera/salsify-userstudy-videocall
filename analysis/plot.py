#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import seaborn as sns
# import pandas as pd

# sns.set(color_codes=True)
# sns.set_style("white")

def plot_delay(xx, yy, o, name):
    plt.clf()
    xxx = sm.add_constant(xx)

    results = sm.OLS(endog=yy, exog=xxx).fit()

    mean, std = [],[]
    r = [1,15,30,60]
    for i in [1,15,30,60]:
        select_y = []
        mean_ = []
        std_ = []
        for k in [10, 14, 18]:
            for j in range(len(xx)):
                if xx[j] == 33*i and (o[j] - k) < 2:
                    select_y.append(yy[j])

            mean_.append( np.mean(select_y) )
            std_.append( np.std(select_y) )

        mean.append(mean_)
        std.append(std_)
            
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
    #plt.xscale('log')
    
    p = None
    centers = [1,15,30,60]
    plt.xticks([33,500,1000,2000])
    for i in range(len(mean)):
        c = 33*centers[i]
        m = mean[i]
        s = std[i]

        count = -66
        for m_, s_ in zip(m,s):
            p = plt.errorbar([c+count], [m_], yerr=[s_],
                             fmt='o', color='k', ecolor='k', capsize=2, capthick=2, lw=2,
                             label='mean ± std')
            count += 66
            
    x = [0, 2050]
    y = [results.params[0] + x[0]*results.params[1], results.params[0] + x[1]*results.params[1]]
    
    #plt.text(34.85, 4.80, 'mean and +/- stddev plotted')

    pp = plt.plot(x,y,'r-',label='regression line')

    #plt.legend()
    patch = mpatches.Patch(color='white', label='R² = ' + str(round(results.rsquared,2)))
    #plt.legend([p,pp[0]], ['mean ± std', 'regression line'])
    plt.legend(handles=[p, pp[0], patch], labels=['mean ± std', '-x/'+str(round(-1/results.params[1],2))+' + ' + str(round(results.params[0],2)), 'R² = ' + str(round(results.rsquared,3))])
    #plt.legend(handles=[p, pp[0], patch], labels=['mean ± std', 'regression line', 'R² = ' + str(round(results.rsquared,3))])
    #    plt.legend([p,pp[0]], ['mean ± std', 'regression line'])


    
    #plt.show()
    plt.savefig(name)
    print(results.params)
    
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
    pp = plt.plot(x,y, 'r', label='regression line')

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

# print('plotting delay')
# x = 33*data[:,1]
# plot_delay(x, y, data[:,3], 'delay.png')
# plot_delay(x, y, data[:,3], 'delay.svg')

#print('plotting quality')
#x = data[:,3]
#plot_quality(x, y, 'quality.png')
#plot_quality(x, y, 'quality.svg')

x = data[:,1:4:2]
x[:,0] = 33*x[:,0] + 250
x = sm.add_constant(x)
results = sm.OLS(endog=y, exog=x).fit()
print(results.params)
print(results.summary())

# groupings
centers = [250,750,1250,2250]

mean, std = [],[]
for delay in [33, 495, 990, 1980]:
    mean_ = []
    std_ = []

    for quality in [10, 14, 18]:
        select_y = []
        for j in range(len(y)):
            if abs(x[j,1] - delay - 250)< 50 and abs(x[j,2] - quality) < 2:
                select_y.append(y[j])
            
        mean_.append( np.mean(select_y) )
        std_.append( np.std(select_y) )
        
    mean.append(mean_)
    std.append(std_)

first = True
ebar = []
padding = [-66,0,66]
for c,m,s in zip(centers, mean, std):
    for p_,m_,s_,q,color in zip(padding, m, s,[10,14,18],['#4c72b0', '#55a868', '#c44e52']):

        plot = plt.errorbar([c+p_], [m_], yerr=[s_],
                         fmt='s', color=color, ecolor=color, capsize=2, capthick=2, lw=2,
                         label='mean ± std')

        if first:
            plt.text(c+p_-40, m_+s_+0.15,str(q), fontsize=9,color=color)

        ebar.append(plot)
        
    if first:
        first = False
        
# lines of best fit
q = [10, 14, 18]
x = [0, 2500]
lines = []
for qq,color in zip(q,['#4c72b0', '#55a868', '#c44e52']):
    y = [results.params[0] + x[0]*results.params[1] + qq*results.params[2], results.params[0] + x[1]*results.params[1] + qq*results.params[2]]
    pp = plt.plot(x, y, color=color,label='regression line')
    lines.append(pp)
    
#print(mean, std)

patch = mpatches.Patch(color='white', label='R² = ' + str(round(results.rsquared,2)))
#plt.legend(handles=[p, lines[1][0], patch], labels=['mean ± std', '-x/'+str(round(-1/results.params[1],2))+' + ' + str(round(results.params[0] + q[1]*results.params[2],2)), 'R² = ' + str(round(results.rsquared,3))])
plt.legend(handles=[ebar[1], lines[1][0], patch], labels=['mean ± std', 'best-fit QoE model', 'R² = ' + str(round(results.rsquared,3))])

# add labels for the groupings
c = 5.6
x = 150
plt.plot([x,x+210],[c,c],lw=1,color=(0.33,0.33,0.33))
plt.plot([x,x],[c,c-.1],lw=1,color=(0.33,0.33,0.33))
plt.plot([x+210,x+210],[c,c-.1],lw=1,color=(0.33,0.33,0.33))

plt.text(x-50, c+.15, 'dB SSIM', fontsize=10, color=(0,0,0))

plt.yticks([1,2,3,4,5])
plt.xticks([250,750,1250,2250])

plt.axis([-100, 2600, 0.75, 6.25])
#plt.title('QoE User Study (Video Call)')
plt.ylabel('QoE score')
plt.xlabel('Video Delay (ms)')
plt.savefig('delay.png')
plt.savefig('delay.svg')

