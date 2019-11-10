import numpy as np
import matplotlib.pyplot as plt

p = .7
n = 20
n_sim = 1000

x = np.arange(n)+1
y = np.arange(n)+1

for loci in range(n):
    loci += 1
    trues = 0
    for i in range(n_sim):
        single_matches = 0
        for i in range(loci):
            if np.sum(np.random.rand(1) < p):
                single_matches += 1
        if single_matches/loci >= .5:
            if single_matches/loci == .5:
                if np.random.rand(1) > .5:
                    trues += 1
            else:
                trues += 1
    y[loci-1] = trues

y=y/n_sim

plt.plot(x, y)
plt.show()