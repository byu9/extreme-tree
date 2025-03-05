import numpy as np

from .validation import ensure_size_at_least


def rank_values(values, against):
    left = np.searchsorted(against, values, side='left')
    right = np.searchsorted(against, values, side='right')
    mid_rank = (left + 1 + right) / 2
    return mid_rank


def empirical_cdf(population):
    population = np.ravel(population)
    cdf_v, counts = np.unique(population, return_counts=True)
    cdf_p = np.cumsum(counts) / len(population)
    return cdf_v, cdf_p


def cramer_von_mises(sample1, sample2):
    sample1 = np.ravel(sample1)
    sample2 = np.ravel(sample2)
    ensure_size_at_least(sample1, 1)
    ensure_size_at_least(sample2, 1)

    values1 = np.sort(sample1)
    values2 = np.sort(sample2)
    combined = np.sort(np.concat([values1, values2]))

    n = len(values1)
    m = len(values2)
    i = 1 + np.arange(n)
    j = 1 + np.arange(m)

    mn = m * n
    m_plus_n = m + n

    r = rank_values(values1, against=combined)
    s = rank_values(values2, against=combined)
    u = np.square(r - i).mean() + np.square(s - j).mean()
    t = u / mn / m_plus_n - (4 * mn - 1) / 6 / m_plus_n

    return t


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

    z = np.sort(np.concat([sample1, sample2]))[..., np.newaxis]
    m1 = np.searchsorted(values1, z, side='right')[:-1].ravel()
    m2 = np.searchsorted(values2, z, side='right')[:-1].ravel()

    term1 = np.square(n * m1 - j * n1) / j / (n - j) / n1
    term2 = np.square(n * m2 - j * n2) / j / (n - j) / n2
    statistic = np.sum(term1 + term2) / n

    return statistic
