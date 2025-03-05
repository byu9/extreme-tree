import numpy as np

from .validation import ensure_size_at_least


def empirical_cdf(population):
    population = np.ravel(population)
    cdf_v, counts = np.unique(population, return_counts=True)
    cdf_p = np.cumsum(counts) / len(population)
    return cdf_v, cdf_p


def empirical_p_values(values, population):
    values = np.reshape(values, shape=(-1, 1))
    cdf_v, cdf_p = empirical_cdf(population)
    p_values = np.interp(values, cdf_v, cdf_p, left=0, right=1)

    return p_values


def kolmogorov_smirnov(sample1, sample2):
    cdf_v1, cdf_p1 = empirical_cdf(sample1)
    cdf_v2, cdf_p2 = empirical_cdf(sample2)

    cdf_v = np.unique(np.concat([cdf_v1, cdf_v2]))
    cdf_p1 = cdf_p1[np.searchsorted(cdf_v1, cdf_v, side='right') - 1]
    cdf_p2 = cdf_p2[np.searchsorted(cdf_v2, cdf_v, side='right') - 1]

    statistic = np.abs(cdf_p1 - cdf_p2).max()
    return statistic


def anderson_darling(sample1, sample2):
    sample1 = np.ravel(sample1)
    sample2 = np.ravel(sample2)

    ensure_size_at_least(sample1, min_size=2)
    ensure_size_at_least(sample2, min_size=2)

    n1 = len(sample1)
    n2 = len(sample2)
    n = n1 + n2
    j = np.arange(1, n)

    values1 = np.unique(sample1)
    values2 = np.unique(sample2)

    z = np.sort(np.concat([sample1, sample2])).reshape(-1, 1)
    m1 = np.searchsorted(values1, z, side='right')[:-1].reshape(-1)
    m2 = np.searchsorted(values2, z, side='right')[:-1].reshape(-1)

    term1 = np.square(n * m1 - j * n1) / j / (n - j) / n1
    term2 = np.square(n * m2 - j * n2) / j / (n - j) / n2
    statistic = np.sum(term1 + term2) / n

    return statistic


def anderson_darling_k(samples):
    samples = [np.ravel(sample) for sample in samples]

    for sample in samples:
        ensure_size_at_least(sample, min_size=2)

    n_k_list = np.array([len(sample) for sample in samples])
    n = n_k_list.sum()
    j = np.arange(1, n)

    values_list = [np.unique(sample) for sample in samples]

    z = np.sort(np.concat(samples)).reshape(-1, 1)
    m_k_list = [
        np.searchsorted(values, z, side='right')[:-1].reshape(-1)
        for values in values_list
    ]

    statistic = sum(
        np.sum(np.square(n * m_k - j * n_k) / j / (n - j)) / n_k
        for m_k, n_k in zip(m_k_list, n_k_list)
    ) / n

    return statistic
