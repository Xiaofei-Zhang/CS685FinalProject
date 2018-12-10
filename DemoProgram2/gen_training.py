import random
import statistics

# generate training data
i = 0

f = open('training_data.txt', 'w')
while i < 100:
  a = random.uniform(0,1)
  b = a**2
  f.write(str(a) + ','  +  str(b) + '\n')
  i += 1


