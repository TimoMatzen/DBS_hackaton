import numpy as np

freqs = [.3, .3, .05, .05, .05, .05, .05, .05, .05, .05]


def generate_alleles():
    return np.random.choice(len(freqs), 2, p=freqs)


def generate_person(allele_pair, mean_rfu, l):
    heights = np.random.gamma(mean_rfu * 10, 1 / 10, 2)
    res = np.zeros((len(freqs, )*l))
    for i, allele in enumerate(allele_pair):
        res[allele] += heights[i]
    return res


def generate_mixture(allele_pairs, mean_rfus, l=1):
    persons = [generate_person(allele_pair, mean_rfu, l) for allele_pair, mean_rfu
               in zip(allele_pairs, mean_rfus)]
    return np.sum(persons, axis=0)


def generate_samples(n, is_binary, n_contributors=3, l=1):
    res = np.zeros((n, len(freqs) * 2 * l))
    for i in range(n):
        allele_pairs = [generate_alleles() for _ in range(2*n_contributors)]
        mean_rfus = np.random.randint(50, 1000, 2*n_contributors)
        res[i, :len(freqs)*l] = generate_mixture(allele_pairs[:n_contributors], mean_rfus[:n_contributors], l)
        if i < n / 2:
            res[i, len(freqs)*l:] = generate_mixture(
                allele_pairs[n_contributors:-1] + [allele_pairs[0]], mean_rfus[n_contributors:], l)
        else:
            res[i, len(freqs)*l:] = generate_mixture(
                allele_pairs[n_contributors:], mean_rfus[n_contributors:], l)
    y = np.array([1] * int(n / 2) + [0] * int(n / 2))
    if is_binary:
        return 1*(res>50), y
    return res, y


if __name__ == '__main__':
    allele_pairs = [generate_alleles() for _ in range(3)]
    X, y  = generate_samples(1000, True, 1)
    binary = 1 * (X > 50)
    print(X)