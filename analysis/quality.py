#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import seaborn as sns
# import pandas as pd

# sns.set(color_codes=True)
# sns.set_style("white")

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'font.family': 'arial'})
f,a = plt.subplots(figsize=(10,5))

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
x[:,0] = 66*x[:,0] + 250
x = sm.add_constant(x)
results = sm.OLS(endog=y, exog=x).fit()
print(results.params)
print(results.summary())

# groupings
delays = [1,15,30,60]

mean, std = [],[]
for quality in [10, 14, 18]:
    mean_ = []
    std_ = []

    for d in delays:
        delay = 66*d + 250
        select_y = []
        for j in range(len(y)):
            if abs(x[j,1] - delay)<10 and abs(x[j,2] - quality) < 2:
                select_y.append(y[j])
            
        mean_.append( np.mean(select_y) )
        std_.append( np.std(select_y) )
        
    mean.append(mean_)
    std.append(std_)

print(mean,std)
    
qualities = [10, 14, 18]
first = True
ebar = []
padding = [-0.375,-0.125,0.125,0.375]
count = 0
for q,m,s in zip(qualities, mean, std):
    c = q
    for p_,m_,s_,d,color in zip(padding, m, s,[300,1200,2200,4200],['#4c72b0', '#55a868', '#c44e52', '#8172b2']):

        plot = plt.errorbar([c+p_], [m_], yerr=[s_],
                         fmt='s', color=color, ecolor=color, capsize=6, capthick=2, lw=3,
                         label='mean ± std')

        if count == 0:
            if d == 300:
                plt.text(c+p_-.195, m_+s_+0.075,str(d), fontsize=15,color=color)
            elif d == 1200:
                plt.text(c+p_-.395, m_-s_-0.29,str(d), fontsize=15,color=color)
            elif d == 2200:
                plt.text(c+p_-.395, m_-s_-0.29,str(d), fontsize=15,color=color)                
            else:
                plt.text(c+p_-.1, m_+s_+0.075,str(d), fontsize=15,color=color)

        ebar.append(plot)

    count += 1
        
    if first:
        first = False
        
# lines of best fit
d = [300,1200,2200,4200]
x = [9.25, 18.75]
#x = [0, 18000]
lines = []
for dd,color in zip(d,['#4c72b0', '#55a868', '#c44e52', '#8172b2']):
    y = [results.params[0] + x[0]*results.params[2] + dd*results.params[1], results.params[0] + x[1]*results.params[2] + dd*results.params[1]]
    pp = plt.plot(x, y, color=color,label='regression line', lw=2)
    lines.append(pp)
    
#print(mean, std)

patch = mpatches.Patch(color='white', label='R² = ' + str(round(results.rsquared,2)))
#plt.legend(handles=[p, lines[1][0], patch], labels=['mean ± std', '-x/'+str(round(-1/results.params[1],2))+' + ' + str(round(results.params[0] + q[1]*results.params[2],2)), 'R² = ' + str(round(results.rsquared,3))])
#plt.legend(handles=[ebar[1], lines[1][0], patch], labels=['mean ± std', 'best-fit QoE model', 'R² = ' + str(round(results.rsquared,3))])
#plt.legend(handles=[ebar[1], lines[1][0]], labels=['mean ± std', 'best-fit QoE model'])

# add labels for the groupings
c = 5.25
x = 9.35
plt.plot([x,x+1.5],[c,c],lw=2,color=(0.33,0.33,0.33))
plt.plot([x,x],[c,c-.1],lw=2,color=(0.33,0.33,0.33))
plt.plot([x+1.5,x+1.5],[c,c-.1],lw=2,color=(0.33,0.33,0.33))

plt.text(x+.75, c+.175, 'Video Delay (ms)', fontsize=15, color=(0,0,0),  horizontalalignment='center',)

plt.yticks([1,2,3,4,5])
#plt.xticks(list(map(lambda x: 66*x+250, [1,15,30,60])))
plt.xticks([10,14,18])

plt.axis([9,19, 0.5, 5.65])
#plt.axis([-100, 20000, -20, 6.25])
#plt.title('QoE User Study (Video Call)')

a.spines['top'].set_visible(False)
a.spines['right'].set_visible(False)
a.get_xaxis().tick_bottom()
a.get_yaxis().tick_left()

plt.ylabel('QoE Score', labelpad=10)
plt.xlabel('Video Quality (SSIM dB)', labelpad=10)
plt.tight_layout()
plt.savefig('quality.png')
plt.savefig('quality.svg')

