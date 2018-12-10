import os
import sys

# Plot the scatter of generated test case for (y=x**2)

epoch_interval = 500
total_epoch = 20000
total = 0
good = 0
for i in range(0,total_epoch, epoch_interval):

    with open(str(i)+'.txt', 'r') as f:
        for line in f:
            a,b = line.split(',')
            # Condition of good test case
            if abs(float(b) - float(a)**2) <= 0.05:
                good += 1
            total += 1

    print("Good test case rate of epoch " + str(i) + " is: %0.2f" % (good/total))

for i in range(0,total_epoch, epoch_interval):
    cmd = 'Rscript plot.r -i ' + str(i)+'.txt -o ' + str(i)+'.png'
    os.system('/bin/bash -c ' + '\"' + cmd + '\"')
