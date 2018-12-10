import random
import statistics

# generate test case for (a<=b<=c)
i = 0

f = open('training_data.txt', 'w')
while i < 100:
  a = random.randint(1,1000)
  b = random.randint(1,1000)
  c = random.randint(1,1000)
  l = [a,b,c]
  f.write(str(min(l)) + ','  + str(statistics.median(l)) + ',' + str(max(l)) + '\n')
  i += 1
