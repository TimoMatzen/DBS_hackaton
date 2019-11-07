import numpy as np

freqs = [.3, .3, .05, .05, .05, .05, .05, .05, .05, .05]


def generate_alleles():
    return np.random.choice(len(freqs), 2, p=freqs)


def generate_person(allele_pair, mean_rfu):
    heights = np.random.gamma(mean_rfu * 10, 1 / 10, 2)
    res = np.zeros((len(freqs, )))
    for i, allele in enumerate(allele_pair):
        res[allele] += heights[i]
    return res


def generate_mixture(allele_pairs, mean_rfus):
    persons = [generate_person(allele_pair, mean_rfu) for allele_pair, mean_rfu
               in zip(allele_pairs, mean_rfus)]
    return np.sum(persons, axis=0)


def generate_samples(n):
    res = np.zeros((n, len(freqs) * 2))
    for i in range(n):
        allele_pairs = [generate_alleles() for _ in range(6)]
        mean_rfus = np.random.randint(50, 1000, 6)
        res[i, :len(freqs)] = generate_mixture(allele_pairs[:3], mean_rfus[:3])
        if i < n / 2:
            res[i, len(freqs):] = generate_mixture(
                allele_pairs[3:5] + [allele_pairs[0]], mean_rfus[3:])
        else:
            res[i, len(freqs):] = generate_mixture(
                allele_pairs[3:], mean_rfus[3:])
    return res, [1] * int(n / 2) + [0] * int(n / 2)


if __name__ == '__main__':
    allele_pairs = [generate_alleles() for _ in range(3)]
    X, y  = generate_samples(1000)
    binary = 1 * (X > 50)
    print(np.mean(np.sum(abs(binary[:, :10]-binary[:, 10:]), axis=1)*y))
    print(np.mean(np.sum(abs(binary[:, :10]-binary[:, 10:]), axis=1)*(1-np.array(y))))
    import matplotlib.pyplot as plt
    plt.matshow(abs(binary[450:550, :10]-binary[450:550, 10:]))
    print(np.mean(binary, axis=0))