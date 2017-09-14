#!/usr/bin/env python3

import json
import os


d = json.loads(open('settings.json', 'r').read())

experiment_time = d['experiment_time']
settings = d['settings']

for setting in settings:
    quantizer = setting['quantizer']
    delay = setting['delay']

    pathname = 'q{}-d{}'.format(quantizer, delay)
    
    with open(os.path.join(pathname, 'ssim1.log'), 'r') as f:
        last_line = f.read().strip().split('\n')[-1]
        y_ssim = float(last_line.split("Y': ")[-1].split('   ')[0])
        
        print(','.join(map(str,[delay, quantizer, y_ssim])))

print('')

for setting in settings:
    quantizer = setting['quantizer']
    delay = setting['delay']

    pathname = 'q{}-d{}'.format(quantizer, delay)
    with open(os.path.join(pathname, 'ssim2.log'), 'r') as f:
        last_line = f.read().strip().split('\n')[-1]
        y_ssim = float(last_line.split("Y': ")[-1].split('   ')[0])
        
        print(','.join(map(str,[delay, quantizer, y_ssim])))

