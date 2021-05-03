import pandas as pd
import numpy as np

n_samples = 1000

d1 = np.random.randint(1, high=7, size=n_samples)
d2 = np.random.randint(1, high=7, size=n_samples)
d3 = np.random.randint(1, high=7, size=n_samples)
d4 = np.random.randint(1, high=7, size=n_samples)

i1 = np.random.randint(1, high=7, size=n_samples)
i2 = np.random.randint(1, high=7, size=n_samples)
i3 = np.random.randint(1, high=7, size=n_samples)

d_sum1 = d1 + d4
d_sum2 = d2 + d3
d_sum3 = d2 + d3

t = np.zeros(n_samples, dtype=np.int8)

for i in range(n_samples):
    dice_sum = d1[i] + d2[i] + d3[i] + d4[i]
    if dice_sum >= 17:
        t[i] = 3
    elif dice_sum >= 14:
        t[i] = 2
    elif dice_sum >= 11:
        t[i] = 1
    else:
        t[i] = 0

dataset = pd.DataFrame()

dataset['d1'] = d1
dataset['d2'] = d2
dataset['d3'] = d3
dataset['d4'] = d4

dataset['i1'] = i1
dataset['i2'] = i2
dataset['i3'] = i3

dataset['d_sum1'] = d_sum1
dataset['d_sum2'] = d_sum2
dataset['d_sum3'] = d_sum3

dataset['Target'] = t

dataset.to_csv('datasets/dice.csv', index=False)
