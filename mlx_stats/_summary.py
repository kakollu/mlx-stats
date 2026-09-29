"""A text summary laid out like statsmodels' OLS summary (same labels, number formats and 78-column layout)."""
import time
import numpy as np


def _forg(x, prec=3):
    x = float(x)
    w = 9 if prec == 3 else 10
    if abs(x) >= 1e4 or abs(x) < 1e-4:
        return f"{x:{w}.{prec}g}"
    return f"{x:{w}.{prec}f}"


def _pair(L, lv, R="", rv="", rw=38):
    left = L + f"{lv:>{max(1, 37 - len(L))}}"
    right = R + f"{rv:>{max(1, rw - len(R))}}" if R else " " * rw
    return f"{left}   {right}"


def summary(res, alpha=0.05):
    m = res.model
    title = "WLS Regression Results" if m._weighted else "OLS Regression Results"
    unc = "" if res.k_constant else " (uncentered)"
    now = time.localtime()
    rows_l = [("Dep. Variable:", m.yname), ("Model:", "WLS" if m._weighted else "OLS"), ("Method:", "Least Squares"),
              ("Date:", time.strftime("%a, %d %b %Y", now)), ("Time:", time.strftime("%H:%M:%S", now)),
              ("No. Observations:", f"{int(res.nobs)}"), ("Df Residuals:", f"{int(res.df_resid)}"),
              ("Df Model:", f"{int(res.df_model)}"), ("Covariance Type:", res.cov_type)]
    rows_r = [("R-squared" + unc + ":", f"{res.rsquared:#8.3f}"), ("Adj. R-squared" + unc + ":", f"{res.rsquared_adj:#8.3f}"),
              ("F-statistic:", f"{res.fvalue:#8.4g}"), ("Prob (F-statistic):", f"{res.f_pvalue:#6.3g}"),
              ("Log-Likelihood:", f"{res.llf:#8.5g}"), ("AIC:", f"{res.aic:#8.4g}"), ("BIC:", f"{res.bic:#8.4g}")]
    rw = max(38, 19 + max(len(R) for R, _ in rows_r))       # statsmodels widens the header table for long labels
    tw = 40 + rw
    pad = tw - len(title)
    out = [" " * (pad - pad // 2) + title + " " * (pad // 2), "=" * tw]     # statsmodels puts the odd space on the left
    for i, (L, lv) in enumerate(rows_l):
        R, rv = rows_r[i] if i < len(rows_r) else ("", "")
        out.append(_pair(L, lv, R, rv, rw))
    # parameter table
    names = res.exog_names
    params, se, t, pv = (np.asarray(v, float) for v in (res.params, res.bse, res.tvalues, res.pvalues))
    ci = np.asarray(res.conf_int(alpha), float)
    stat, ptag = ("t", "P>|t|") if res.use_t else ("z", "P>|z|")
    stub = max(10, max(len(n) for n in names))
    widths = [11, 11, 11, 11, 12, 12]
    head = ["coef", "std err", stat, ptag, f"[{alpha / 2:g}", f"{1 - alpha / 2:g}]"]
    width = stub + sum(widths)
    out.append("=" * width)
    out.append(" " * stub + "".join(f"{h:>{w}}" for h, w in zip(head, widths)))
    out.append("-" * width)
    for i, nm in enumerate(names):
        cells = [_forg(params[i], 4), _forg(se[i]), _forg(t[i]), f"{pv[i]:#6.3f}", _forg(ci[i, 0]), _forg(ci[i, 1])]
        out.append(f"{nm:<{stub}}" + "".join(f"{c.strip():>{w}}" for c, w in zip(cells, widths)))
    out.append("=" * width)
    dg = res.diagn
    if dg is not None:
        out.append(_pair("Omnibus:", f"{dg['omnibus']:#6.3f}", "Durbin-Watson:", f"{dg['durbin_watson']:#8.3f}"))
        out.append(_pair("Prob(Omnibus):", f"{dg['omnibus_pvalue']:#6.3f}", "Jarque-Bera (JB):", f"{dg['jarque_bera']:#8.3f}"))
        out.append(_pair("Skew:", f"{dg['skew']:#6.3f}", "Prob(JB):", f"{dg['jarque_bera_pvalue']:#8.3g}"))
        out.append(_pair("Kurtosis:", f"{dg['kurtosis']:#6.3f}", "Cond. No.", f"{res.condition_number:#8.3g}"))
        out.append("=" * 78)
    else:
        out.append(_pair("Cond. No.", f"{res.condition_number:#8.3g}"))
        out.append("=" * 78)
    notes = []
    if not res.k_constant:
        notes.append("R² is computed without centering (uncentered) since the model does not contain a constant.")
    if res.cov_type == "nonrobust":
        notes.append("Standard Errors assume that the covariance matrix of the errors is correctly specified.")
    else:
        notes.append(f"Standard Errors are heteroscedasticity robust ({res.cov_type})")
    ev = res.eigenvals
    if ev[-1] < 1e-10:
        notes.append(f"The smallest eigenvalue is {ev[-1]:6.3g}. This might indicate that there are\n"
                     "strong multicollinearity problems or that the design matrix is singular.")
    elif res.condition_number > 1000:
        notes.append(f"The condition number is large, {res.condition_number:#8.3g}. This might indicate that there are\n"
                     "strong multicollinearity or other numerical problems.")
    notes.append(f"Computed on the GPU in {res._passes} passes over the data ({res.fit_seconds:.3f} s); float32 data, float64 statistics.")
    out.append("")
    out.append("Notes:")
    for i, nt in enumerate(notes, 1):
        out.append(f"[{i}] {nt}")
    return Summary("\n".join(out))


class Summary:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    __repr__ = __str__

    def as_text(self):
        return self.text
