import os
import sys

epoch_interval = 500
total_epoch = 20000
total = 0
good = 0
for i in range(0,total_epoch, epoch_interval):

    with open(str(i)+'.txt', 'r') as f:
        for line in f:
            a,b,c = line.split(',')
            if a <= b and b <= c:
                good += 1
            total += 1

    print("Good test case rate of epoch " + str(i) + " is: %0.2f" % (good/total))


