"""Normality tests from residual power sums: the same formulas as scipy.stats.skew/kurtosis/skewtest/kurtosistest/
normaltest and statsmodels' jarque_bera, but taking sums instead of the n residuals (n can be 10^9 here)."""
import numpy as np
from scipy import stats


def central_moments(s1, s2, s3, s4, n):
    mu = s1 / n
    m2 = s2 / n - mu * mu
    m3 = s3 / n - 3 * mu * s2 / n + 2 * mu ** 3
    m4 = s4 / n - 4 * mu * s3 / n + 6 * mu * mu * s2 / n - 3 * mu ** 4
    return m2, m3, m4


def skew_kurt(m2, m3, m4):
    """biased (population) skewness and Pearson kurtosis, as scipy.stats.skew / kurtosis(fisher=False)."""
    if m2 <= 0:
        return np.nan, np.nan
    return m3 / m2 ** 1.5, m4 / m2 ** 2


def skewtest_z(b2, n):
    y = b2 * np.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    beta2 = (3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9)))
    W2 = -1 + np.sqrt(2 * (beta2 - 1))
    delta = 1 / np.sqrt(0.5 * np.log(W2))
    alpha = np.sqrt(2.0 / (W2 - 1))
    y = 1.0 if y == 0 else y
    return delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))


def kurtosistest_z(b2, n):
    E = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.) * (n + 3) * (n + 5))
    x = (b2 - E) / np.sqrt(varb2)
    sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * np.sqrt((6.0 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
    A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + np.sqrt(1 + 4.0 / (sqrtbeta1 ** 2)))
    term1 = 1 - 2 / (9.0 * A)
    denom = 1 + x * np.sqrt(2 / (A - 4.0))
    if denom == 0:
        return np.nan
    term2 = np.sign(denom) * ((1 - 2.0 / A) / abs(denom)) ** (1 / 3.0)
    return (term1 - term2) / np.sqrt(2 / (9.0 * A))


def diagnostics(s1, s2, s3, s4, n, dw_num):
    m2, m3, m4 = central_moments(s1, s2, s3, s4, n)
    skew, kurt = skew_kurt(m2, m3, m4)
    out = {"skew": skew, "kurtosis": kurt}
    jb = (n / 6.0) * (skew ** 2 + 0.25 * (kurt - 3) ** 2)
    out["jarque_bera"] = jb
    out["jarque_bera_pvalue"] = stats.chi2.sf(jb, 2)
    if n >= 8:
        k2 = skewtest_z(skew, n) ** 2 + kurtosistest_z(kurt, n) ** 2
        out["omnibus"] = k2
        out["omnibus_pvalue"] = stats.chi2.sf(k2, 2)
    else:
        out["omnibus"] = out["omnibus_pvalue"] = np.nan
    out["durbin_watson"] = dw_num / s2 if s2 > 0 else np.nan
    return out
