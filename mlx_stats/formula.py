"""statsmodels-style formulas: ms.ols("price ~ size + age + C(region)", data=df).fit()

The design matrix is built with formulaic, the engine statsmodels 0.15 uses, with statsmodels' term order and its default
missing-value rule (rows with a missing value in any variable of the formula are dropped). Parameter names and order are
therefore those `statsmodels.formula.api.ols` gives for the same formula and data.
"""
import sys
from collections import defaultdict
import numpy as np


def _legacy_formula(formula, data, context):
    """statsmodels' legacy term order: intercept, then all-categorical terms by degree, then the rest grouped by their
    continuous factors. Ported from statsmodels/formula/_manager.py (BSD-3-Clause, the statsmodels developers)."""
    import formulaic
    mm = formulaic.model_matrix(formula, data, context=context)
    flags = formulaic.parser.DefaultFormulaParser.FeatureFlags.TWOSIDED
    parser = formulaic.parser.DefaultFormulaParser(feature_flags=flags)
    f = formulaic.Formula(formula, _parser=parser, _ordering="none")
    if isinstance(mm, formulaic.ModelMatrices):
        rhs, rhs_f, lhs_f = mm.rhs, f.rhs, f.lhs
    else:
        rhs, rhs_f, lhs_f = mm, f, None
    has_intercept = any(t.degree == 0 and str(t) == "1" for t in rhs_f)
    parser = formulaic.parser.DefaultFormulaParser(feature_flags=flags, include_intercept=has_intercept)
    terms = list(parser.get_terms(str(rhs_f)).root)
    cat_vars = list(rhs.model_spec.factor_contrasts.keys())
    intercept = [t for t in terms if t.degree == 0]
    rest = [t for t in terms if t.degree != 0]
    cats = sorted([t for t in rest if all(fc in cat_vars for fc in t.factors)], key=lambda t: t.degree)
    rest = [t for t in rest if t not in cats]
    groups = defaultdict(list)
    for t in rest:
        groups[tuple(sorted(fc for fc in t.factors if fc not in cat_vars))].append(t)
    conts = [t for g in groups.values() for t in sorted(g, key=lambda t: t.degree)]
    rhs_new = formulaic.Formula(intercept + cats + conts, _ordering="none")
    if lhs_f is None:
        return rhs_new
    return formulaic.Formula(lhs=formulaic.Formula(lhs_f, _ordering="none"), rhs=rhs_new, _ordering="none")


def _caller_context(depth):
    frame = sys._getframe(depth + 1)
    ctx = {"np": np}
    ctx.update(frame.f_globals)
    ctx.update(frame.f_locals)
    return ctx


def design(formula, data, context=None):
    """-> (y: pandas Series, X: pandas DataFrame) for a two-sided formula, rows with missing values dropped."""
    import formulaic
    import pandas as pd
    if "~" not in formula:
        raise ValueError("the formula needs a response: 'y ~ x1 + x2'")
    if not hasattr(data, "columns"):
        data = pd.DataFrame(data)
    ctx = context if context is not None else _caller_context(2)
    f = _legacy_formula(formula, data, ctx)
    mm = formulaic.model_matrix(f, data, context=ctx, na_action="drop")
    lhs, rhs = mm.lhs, mm.rhs
    if lhs.shape[1] != 1:
        raise ValueError("the formula's left-hand side must be a single variable")
    y = pd.Series(np.asarray(lhs.iloc[:, 0], dtype=np.float64), index=lhs.index, name=lhs.columns[0])
    X = pd.DataFrame(np.asarray(rhs, dtype=np.float64), index=rhs.index, columns=list(rhs.columns))
    return y, X


def ols(formula, data, **kwargs):
    """OLS model from a formula and a DataFrame (or dict of columns): ms.ols("y ~ x + C(g)", df).fit()."""
    from .regression import OLS
    y, X = design(formula, data, _caller_context(1))
    return OLS(y, X, **kwargs)


def wls(formula, data, weights, **kwargs):
    """WLS model from a formula; weights: an array aligned with data's rows, or the name of a column of data."""
    import pandas as pd
    from .regression import WLS
    y, X = design(formula, data, _caller_context(1))
    if isinstance(weights, str):
        w = data[weights]
    else:
        w = pd.Series(np.asarray(weights, dtype=np.float64).reshape(-1), index=getattr(data, "index", None))
    return WLS(y, X, weights=w.loc[X.index].to_numpy(dtype=np.float64), **kwargs)
