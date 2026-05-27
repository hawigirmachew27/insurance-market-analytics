"""
src/hypothesis_tests.py
=======================
Statistical test wrappers for Task 3 – A/B Hypothesis Testing.

For each of the four null hypotheses the caller:
  1. Segments the data into Group A (control) and Group B (test).
  2. Picks the right KPI (claim frequency, claim severity, or margin).
  3. Calls the appropriate test function.
  4. Reads the returned dict for the p-value and decision.

Functions
---------
compute_claim_frequency   – Fraction of policies with TotalClaims > 0.
compute_claim_severity    – Mean TotalClaims among policies that had a claim.
run_chi_squared           – Chi-squared test for categorical/frequency KPIs.
run_t_test                – Independent-samples t-test for numerical KPIs.
run_z_test                – Two-proportion z-test for large samples.
test_province_risk        – H0: no risk difference across provinces.
test_zipcode_risk         – H0: no risk difference between zip codes.
test_zipcode_margin       – H0: no margin difference between zip codes.
test_gender_risk          – H0: no risk difference between Women and Men.
summarise_results         – Collect all four results into a tidy table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ─── KPI helpers ──────────────────────────────────────────────────────────────

def compute_claim_frequency(group: pd.Series) -> float:
    """
    Proportion of policies in ``group`` that had at least one claim.

    Claim frequency = number of policies with TotalClaims > 0
                      divided by total policies in the group.

    Parameters
    ----------
    group : pd.Series
        TotalClaims values for the group.

    Returns
    -------
    float  in [0, 1]
    """
    group = pd.to_numeric(group, errors="coerce").dropna()
    if len(group) == 0:
        return np.nan
    return (group > 0).mean()


def compute_claim_severity(group: pd.Series) -> pd.Series:
    """
    Return the TotalClaims values for policies that had a claim (> 0).

    Used as the input series for a t-test on claim severity.

    Parameters
    ----------
    group : pd.Series
        TotalClaims values for the group.

    Returns
    -------
    pd.Series  containing only positive claim amounts.
    """
    group = pd.to_numeric(group, errors="coerce").dropna()
    return group[group > 0]


# ─── Statistical tests ────────────────────────────────────────────────────────

def run_chi_squared(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-squared test comparing claim frequency between two groups.

    Builds a 2×2 contingency table:
        rows    = Group A / Group B
        columns = claim / no claim

    A chi-squared test is appropriate here because the KPI is a
    proportion (categorical: did a claim occur or not?).

    Parameters
    ----------
    group_a : pd.Series
        TotalClaims for Group A (control).
    group_b : pd.Series
        TotalClaims for Group B (test).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        test, statistic, p_value, alpha, reject_h0,
        freq_a, freq_b, n_a, n_b
    """
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()

    # Contingency table: [claims > 0, claims == 0] for each group
    a_claim    = (a > 0).sum()
    a_no_claim = (a == 0).sum()
    b_claim    = (b > 0).sum()
    b_no_claim = (b == 0).sum()

    table = np.array([[a_claim, a_no_claim],
                      [b_claim, b_no_claim]])

    chi2, p_value, _, _ = stats.chi2_contingency(table)

    return {
        "test":       "chi-squared",
        "statistic":  round(chi2, 4),
        "p_value":    round(p_value, 6),
        "alpha":      alpha,
        "reject_h0":  p_value < alpha,
        "freq_a":     round((a > 0).mean(), 4),
        "freq_b":     round((b > 0).mean(), 4),
        "n_a":        len(a),
        "n_b":        len(b),
    }


def run_t_test(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Independent-samples (Welch's) t-test comparing a numerical KPI.

    Welch's t-test is used instead of Student's because it does not
    assume equal variances between the two groups — appropriate for
    insurance data where claim distributions can differ dramatically.

    Parameters
    ----------
    group_a : pd.Series
        Numerical KPI values for Group A (control).
    group_b : pd.Series
        Numerical KPI values for Group B (test).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        test, statistic, p_value, alpha, reject_h0,
        mean_a, mean_b, n_a, n_b
    """
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    return {
        "test":      "Welch t-test",
        "statistic": round(t_stat, 4),
        "p_value":   round(p_value, 6),
        "alpha":     alpha,
        "reject_h0": p_value < alpha,
        "mean_a":    round(a.mean(), 4),
        "mean_b":    round(b.mean(), 4),
        "n_a":       len(a),
        "n_b":       len(b),
    }


def run_z_test(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Two-proportion z-test comparing claim frequency in large samples.

    A z-test is valid when both groups are large (n > 30) and the KPI
    is a proportion. It is an alternative to chi-squared for frequency
    comparisons when the sampling distribution is approximately normal.

    Parameters
    ----------
    group_a : pd.Series
        TotalClaims for Group A (control).
    group_b : pd.Series
        TotalClaims for Group B (test).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        test, statistic, p_value, alpha, reject_h0,
        freq_a, freq_b, n_a, n_b
    """
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()

    p_a, n_a = (a > 0).mean(), len(a)
    p_b, n_b = (b > 0).mean(), len(b)

    # Pooled proportion under H0
    p_pool = ((a > 0).sum() + (b > 0).sum()) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    z_stat  = (p_a - p_b) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))   # two-tailed

    return {
        "test":      "two-proportion z-test",
        "statistic": round(z_stat, 4),
        "p_value":   round(p_value, 6),
        "alpha":     alpha,
        "reject_h0": p_value < alpha,
        "freq_a":    round(p_a, 4),
        "freq_b":    round(p_b, 4),
        "n_a":       n_a,
        "n_b":       n_b,
    }


# ─── Hypothesis-specific runners ─────────────────────────────────────────────

def test_province_risk(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    H0: There are no risk differences across provinces.

    KPI: Claim Severity (average TotalClaims among claimants).
    Method: Welch t-test on the two highest-volume provinces.

    The two largest provinces by policy count are selected as the
    comparison pair so the test has maximum statistical power.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Province' and 'TotalClaims'.
    alpha : float

    Returns
    -------
    dict  (result dict from run_t_test plus 'province_a' and 'province_b')
    """
    col_prov  = _find_col(df, "Province")
    col_claim = _find_col(df, "TotalClaims")

    # Pick the two largest provinces by policy count
    top2 = (
        df[col_prov].value_counts()
        .head(2)
        .index.tolist()
    )
    prov_a, prov_b = top2[0], top2[1]

    sev_a = compute_claim_severity(df.loc[df[col_prov] == prov_a, col_claim])
    sev_b = compute_claim_severity(df.loc[df[col_prov] == prov_b, col_claim])

    result = run_t_test(sev_a, sev_b, alpha=alpha)
    result.update({
        "hypothesis": "H0: No risk difference across provinces",
        "kpi":        "Claim Severity",
        "group_a":    prov_a,
        "group_b":    prov_b,
    })
    return result


def test_zipcode_risk(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    H0: There are no risk differences between zip codes.

    KPI: Claim Frequency (proportion of policies with a claim).
    Method: Chi-squared test on the two highest-volume postal codes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'PostalCode' and 'TotalClaims'.
    alpha : float

    Returns
    -------
    dict
    """
    col_zip   = _find_col(df, "PostalCode")
    col_claim = _find_col(df, "TotalClaims")

    top2 = df[col_zip].value_counts().head(2).index.tolist()
    zip_a, zip_b = top2[0], top2[1]

    freq_a = df.loc[df[col_zip] == zip_a, col_claim]
    freq_b = df.loc[df[col_zip] == zip_b, col_claim]

    result = run_chi_squared(freq_a, freq_b, alpha=alpha)
    result.update({
        "hypothesis": "H0: No risk difference between zip codes",
        "kpi":        "Claim Frequency",
        "group_a":    zip_a,
        "group_b":    zip_b,
    })
    return result


def test_zipcode_margin(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    H0: There is no significant margin difference between zip codes.

    KPI: Margin = TotalPremium − TotalClaims.
    Method: Welch t-test on the two highest-volume postal codes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'PostalCode', 'TotalPremium', and 'TotalClaims'.
    alpha : float

    Returns
    -------
    dict
    """
    col_zip   = _find_col(df, "PostalCode")
    col_prem  = _find_col(df, "TotalPremium")
    col_claim = _find_col(df, "TotalClaims")

    df = df.copy()
    df["_Margin"] = (
        pd.to_numeric(df[col_prem],  errors="coerce") -
        pd.to_numeric(df[col_claim], errors="coerce")
    )

    top2  = df[col_zip].value_counts().head(2).index.tolist()
    zip_a, zip_b = top2[0], top2[1]

    margin_a = df.loc[df[col_zip] == zip_a, "_Margin"].dropna()
    margin_b = df.loc[df[col_zip] == zip_b, "_Margin"].dropna()

    result = run_t_test(margin_a, margin_b, alpha=alpha)
    result.update({
        "hypothesis": "H0: No margin difference between zip codes",
        "kpi":        "Margin (TotalPremium − TotalClaims)",
        "group_a":    zip_a,
        "group_b":    zip_b,
    })
    return result


def test_gender_risk(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    H0: There is no significant risk difference between Women and Men.

    KPI: Claim Severity (average TotalClaims among claimants).
    Method: Welch t-test on Male vs Female policies.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Gender' and 'TotalClaims'.
    alpha : float

    Returns
    -------
    dict
    """
    col_gender = _find_col(df, "Gender")
    col_claim  = _find_col(df, "TotalClaims")

    # Normalise gender strings to handle "Male"/"male"/"M" variations
    gender_series = df[col_gender].astype(str).str.strip().str.lower()

    male_claims   = compute_claim_severity(
        df.loc[gender_series.str.contains("male") & ~gender_series.str.contains("fe"), col_claim]
    )
    female_claims = compute_claim_severity(
        df.loc[gender_series.str.contains("fe"), col_claim]
    )

    result = run_t_test(male_claims, female_claims, alpha=alpha)
    result.update({
        "hypothesis": "H0: No risk difference between Women and Men",
        "kpi":        "Claim Severity",
        "group_a":    "Male",
        "group_b":    "Female",
    })
    return result


# ─── Summary table ────────────────────────────────────────────────────────────

def summarise_results(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run all four hypothesis tests and return a tidy summary table.

    Parameters
    ----------
    df : pd.DataFrame
        The full insurance DataFrame (output of load_data + engineer_base_features).
    alpha : float
        Significance level applied to all tests (default 0.05).

    Returns
    -------
    pd.DataFrame with columns:
        hypothesis, kpi, group_a, group_b, test, statistic,
        p_value, reject_h0, decision, business_interpretation
    """
    results = [
        test_province_risk(df,  alpha),
        test_zipcode_risk(df,   alpha),
        test_zipcode_margin(df, alpha),
        test_gender_risk(df,    alpha),
    ]

    rows = []
    for r in results:
        decision = "Reject H₀" if r["reject_h0"] else "Fail to reject H₀"
        rows.append({
            "hypothesis":  r["hypothesis"],
            "kpi":         r["kpi"],
            "group_a":     r.get("group_a", ""),
            "group_b":     r.get("group_b", ""),
            "test":        r["test"],
            "statistic":   r["statistic"],
            "p_value":     r["p_value"],
            "reject_h0":   r["reject_h0"],
            "decision":    decision,
        })

    return pd.DataFrame(rows)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, name: str) -> str:
    """Case-insensitive column lookup. Raises KeyError if not found."""
    for col in df.columns:
        if col.strip().lower() == name.lower():
            return col
    raise KeyError(f"Column '{name}' not found in DataFrame. Available: {df.columns.tolist()}")
