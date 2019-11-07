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


def generate_samples(n, is_binary, n_contributors=3, threshold=50, n_samples=2):
    res = np.zeros((n, len(freqs) * n_samples))
    for i in range(n):
        allele_pairs = [generate_alleles() for _ in range(n_samples*n_contributors)]
        mean_rfus = np.random.randint(50, 1000, n_samples*n_contributors)
        res[i, :len(freqs)] = generate_mixture(allele_pairs[:n_contributors], mean_rfus[:n_contributors])
        if i < n / 2:
            res[i, len(freqs):len(freqs)*2] = generate_mixture(
                allele_pairs[n_contributors:2*n_contributors-1] + [allele_pairs[0]], mean_rfus[n_contributors:2*n_contributors])
            res[i, len(freqs)*2:] = generate_mixture(
                allele_pairs[n_contributors*2:-1] + [allele_pairs[0]], mean_rfus[2*n_contributors:])
        else:
            res[i, len(freqs):len(freqs)*2] = generate_mixture(
                allele_pairs[n_contributors:2*n_contributors], mean_rfus[n_contributors:2*n_contributors])
            res[i, len(freqs)*2:] = generate_mixture(
                allele_pairs[2*n_contributors:], mean_rfus[2*n_contributors:])
    y = np.array([1] * int(n / 2) + [0] * int(n / 2))
    if is_binary:
        return 1*(res>threshold), y
    return res, y


if __name__ == '__main__':
    allele_pairs = [generate_alleles() for _ in range(3)]
    X, y  = generate_samples(10, False, 1, n_samples=3)
    binary = 1 * (X > 50)
    print(X.shape)