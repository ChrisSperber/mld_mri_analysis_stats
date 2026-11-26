"""Linear mixed-effects models and statistical comparison of nested models.

Provides functions for random-intercept models, Nakagawa-style marginal R²,
likelihood-ratio tests, and subject-level bootstrap confidence intervals
for incremental marginal R².
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper

TOLERANCE_LRT = 1e-6


@dataclass
class MixedMarkerResult:
    """Structured output of mixed model analysis."""

    n_subjects: int
    n_sessions: int

    r2_null: float
    r2_full: float
    r2_delta: float

    ll_null: float
    ll_full: float

    lrt_stat: float
    lrt_df: int
    lrt_pvalue: float

    beta_marker: float
    beta_marker_se: float

    r2_delta_ci_low: float | None = None
    r2_delta_ci_high: float | None = None
    n_bootstrap: int = 0
    n_bootstrap_valid: int = 0


def fit_mixed_model(
    df: pd.DataFrame,
    formula: str,
    subject_colname: str,
) -> MixedLMResultsWrapper:
    """Fit a linear mixed model with a subject-specific random intercept.

    Models are fitted using maximum likelihood (ML), rather than restricted
    maximum likelihood (REML), to allow likelihood-ratio tests between
    models differing in their fixed effects.

    Raises an error if the model does not converge or the resulting
    log-likelihood is non-finite.
    """
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df[subject_colname],
        re_formula="1",
        missing="raise",
    )

    result = model.fit(reml=False)

    if not result.converged:
        raise RuntimeError(f"Mixed model did not converge: {formula}")

    if not np.isfinite(result.llf):
        raise RuntimeError(f"Non-finite log-likelihood: {formula}")

    return result


def marginal_r2(result: MixedLMResultsWrapper) -> float:
    """Compute Nakagawa-style marginal R² for a random-intercept mixed model.

    R²_marg = Var(fixed) / (Var(fixed) + Var(random) + Var(residual))

    Marginal R² quantifies the variance explained by the fixed effects,
    as opposed to conditional R², which also includes the random effects.

    Assumes:
        - Random intercept only.
        - Gaussian residuals.
        - No additional variance components.
    """
    # Fixed-effects design matrix and parameters
    exog_fe = result.model.exog
    beta_fe = result.fe_params

    # Fixed-effects linear predictor
    eta_fe = np.dot(exog_fe, beta_fe)
    var_fe = np.var(eta_fe, ddof=1)

    # Variance of the random intercept (cov_re is a 1 x 1 matrix)
    var_re = float(result.cov_re.iloc[0, 0])

    # Residual variance
    var_resid = float(result.scale)

    denominator = var_fe + var_re + var_resid

    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Invalid variance components for marginal R².")

    return float(var_fe / denominator)


def likelihood_ratio_test(
    result_null: MixedLMResultsWrapper,
    result_full: MixedLMResultsWrapper,
) -> tuple[float, int, float]:
    """Perform a likelihood-ratio test between two nested mixed models.

    LR = 2 * (LL_full - LL_null)

    The reference distribution is chi-squared, with degrees of freedom
    corresponding to the difference in the number of model parameters.

    Assumes:
        - Both models were fitted using maximum likelihood.
        - Both models were fitted on identical observations.
        - The reduced model is nested within the full model.
        - The random-effects structure is identical.

    Returns
    -------
    tuple
        Likelihood-ratio statistic, degrees of freedom, and p-value.

    """
    lr_stat = 2.0 * (result_full.llf - result_null.llf)
    df_diff = int(result_full.df_modelwc - result_null.df_modelwc)

    if df_diff <= 0:
        raise ValueError("Full model must contain more parameters than null model.")

    if lr_stat < -TOLERANCE_LRT:
        raise RuntimeError("Full model has a lower log-likelihood than the null model.")

    # Allow negligible negative values due to numerical precision
    lr_stat = max(0.0, lr_stat)

    pvalue = float(chi2.sf(lr_stat, df=df_diff))

    return lr_stat, df_diff, pvalue


def resample_subjects(
    df: pd.DataFrame,
    subject_colname: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create a subject-level bootstrap sample.

    Subjects are sampled with replacement, retaining all observations
    belonging to each sampled subject.

    Each sampled copy receives a new subject identifier. Thus, if a subject
    is sampled multiple times, its copies are treated as independent
    clusters rather than additional measurements of the same subject.
    """
    subjects = df[subject_colname].unique()
    subjects = np.sort(subjects)

    sampled_subjects = rng.choice(
        subjects,
        size=len(subjects),
        replace=True,
    )

    bootstrap_samples = []

    for bootstrap_id, subject in enumerate(sampled_subjects):
        subject_data = df[df[subject_colname] == subject].copy()
        subject_data[subject_colname] = bootstrap_id

        bootstrap_samples.append(subject_data)

    return pd.concat(bootstrap_samples, ignore_index=True)


def bootstrap_delta_r2(  # noqa: PLR0913, PLR0917
    df: pd.DataFrame,
    formula_null: str,
    formula_full: str,
    subject_colname: str,
    n_bootstrap: int,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float | None, float | None, int]:
    """Estimate a percentile bootstrap confidence interval for incremental R².

    Resampling is performed at subject level rather than observation level
    to preserve the dependence structure of repeated measurements.

    Both the null and full model are refitted on each bootstrap sample.

    Bootstrap iterations with fitting or convergence failures are omitted.
    The number of valid iterations is returned for quality control.

    Parameters
    ----------
    df : pd.DataFrame
        One row per measurement.
    formula_null : str
        Formula of the reduced model.
    formula_full : str
        Formula of the full model.
    subject_colname : str
        Column containing subject identifiers.
    n_bootstrap : int
        Number of attempted bootstrap iterations.
    ci_level : float
        Confidence level, defaults to 0.95.
    random_state : int
        Random seed for reproducible subject-level resampling.

    Returns
    -------
    tuple
        Lower confidence limit, upper confidence limit, and number of
        successful bootstrap iterations.

    """
    if n_bootstrap <= 0:
        raise ValueError("Number of bootstrap iterations must be positive.")

    if not 0 < ci_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    r2_delta_boot = []

    for _ in range(n_bootstrap):
        boot_df = resample_subjects(
            df,
            subject_colname,
            rng,
        )

        try:
            boot_null = fit_mixed_model(
                boot_df,
                formula_null,
                subject_colname,
            )

            boot_full = fit_mixed_model(
                boot_df,
                formula_full,
                subject_colname,
            )

            boot_r2_null = marginal_r2(boot_null)
            boot_r2_full = marginal_r2(boot_full)

            r2_delta_boot.append(boot_r2_full - boot_r2_null)

        except (RuntimeError, np.linalg.LinAlgError):
            # Skip bootstrap samples with numerical or convergence failures
            continue

    n_valid = len(r2_delta_boot)

    if n_valid < n_bootstrap:
        warnings.warn(
            f"Only {n_valid} of {n_bootstrap} bootstrap iterations were valid.",
            RuntimeWarning,
            stacklevel=2,
        )

    if n_valid < 2:  # noqa: PLR2004
        return None, None, n_valid

    alpha = 1.0 - ci_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    ci_low = float(np.quantile(r2_delta_boot, lower_q))
    ci_high = float(np.quantile(r2_delta_boot, upper_q))

    return ci_low, ci_high, n_valid


def fit_marker_mixed_model(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    marker_colname: str,
    target_var_colname: str,
    subject_colname: str,
    covariate_colnames: list[str] | None = None,
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> MixedMarkerResult:
    """Fit and compare two linear mixed models with random subject intercepts.

    Null: target ~ covariates + (1 | subject)
    Full: target ~ covariates + marker + (1 | subject)

    Returns marginal R² for both models, incremental marginal R²,
    likelihood-ratio test results, and the marker regression coefficient.

    Optionally computes a subject-level bootstrap confidence interval
    for incremental marginal R².

    Parameters
    ----------
    df : pd.DataFrame
        One row per measurement, with repeated measurements on different
        rows.
    marker_colname : str
        Column containing the continuous marker of interest.
    target_var_colname : str
        Column containing the outcome variable.
    subject_colname : str
        Column containing subject IDs (random-intercept grouping factor).
    covariate_colnames : list[str] or None
        Columns included as fixed effects in both models.
        If None or empty, the baseline model contains only an intercept
        and the random subject intercept.
    n_bootstrap : int
        Number of bootstrap iterations for the incremental R² confidence
        interval. Defaults to 0, disabling bootstrapping.
    ci_level : float
        Confidence level, defaults to 0.95.
    random_state : int
        Random seed for reproducible bootstrapping.

    Returns
    -------
    MixedMarkerResult
        Contains sample sizes, marginal R² values, log-likelihoods,
        LRT statistic, degrees of freedom, p-value, marker coefficient
        and standard error, and optional bootstrap confidence limits.

    """
    if covariate_colnames is None:
        covariate_colnames = []

    if n_bootstrap < 0:
        raise ValueError("Number of bootstrap iterations cannot be negative.")

    # Select relevant columns and ensure identical samples for both models
    needed_cols = [
        subject_colname,
        target_var_colname,
        marker_colname,
        *covariate_colnames,
    ]

    df_model = df[needed_cols].copy()

    if df_model.isna().any().any():
        missing = df_model.isna().sum()
        raise ValueError(f"Missing values in model data:\n{missing[missing > 0]}")

    # Validate data types of outcome and marker
    for column in [target_var_colname, marker_colname]:
        if not pd.api.types.is_numeric_dtype(df_model[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

    # Covariates must be numeric or explicitly categorical
    for column in covariate_colnames:
        is_numeric = pd.api.types.is_numeric_dtype(df_model[column])
        is_categorical = isinstance(df_model[column].dtype, pd.CategoricalDtype)

        if not (is_numeric or is_categorical):
            raise ValueError(
                f"Covariate '{column}' must be numeric or explicitly categorical. "
                f"Found dtype: {df_model[column].dtype}."
            )

    # Construct model formulas
    baseline_terms = " + ".join(covariate_colnames) or "1"

    formula_null = f"{target_var_colname} ~ {baseline_terms}"
    formula_full = f"{formula_null} + {marker_colname}"

    # Fit both models using maximum likelihood
    result_null = fit_mixed_model(
        df_model,
        formula_null,
        subject_colname,
    )

    result_full = fit_mixed_model(
        df_model,
        formula_full,
        subject_colname,
    )

    # Marginal R² of each model and incremental explanatory value
    r2_null = marginal_r2(result_null)
    r2_full = marginal_r2(result_full)
    r2_delta = r2_full - r2_null

    # Likelihood-ratio test of the additional marker term
    lrt_stat, lrt_df, lrt_pvalue = likelihood_ratio_test(
        result_null,
        result_full,
    )

    # Marker coefficient and Wald standard error from the full model
    beta_marker = float(result_full.fe_params[marker_colname])
    beta_marker_se = float(result_full.bse_fe[marker_colname])

    # Optional bootstrap confidence interval for incremental R²
    ci_low = None
    ci_high = None
    n_bootstrap_valid = 0

    if n_bootstrap > 0:
        ci_low, ci_high, n_bootstrap_valid = bootstrap_delta_r2(
            df_model,
            formula_null,
            formula_full,
            subject_colname,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            random_state=random_state,
        )

    return MixedMarkerResult(
        n_subjects=df_model[subject_colname].nunique(),
        n_sessions=len(df_model),
        r2_null=r2_null,
        r2_full=r2_full,
        r2_delta=r2_delta,
        ll_null=float(result_null.llf),
        ll_full=float(result_full.llf),
        lrt_stat=lrt_stat,
        lrt_df=lrt_df,
        lrt_pvalue=lrt_pvalue,
        beta_marker=beta_marker,
        beta_marker_se=beta_marker_se,
        r2_delta_ci_low=ci_low,
        r2_delta_ci_high=ci_high,
        n_bootstrap=n_bootstrap,
        n_bootstrap_valid=n_bootstrap_valid,
    )
