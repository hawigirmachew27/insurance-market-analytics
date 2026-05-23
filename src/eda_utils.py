"""
eda_utils.py
============
Reusable helper functions for Exploratory Data Analysis (EDA) on the ACIS
insurance dataset.

Each function is designed to be called from the EDA notebook
(``notebooks/01_eda.ipynb``) and returns either a ``matplotlib`` Figure or a
summary ``pd.DataFrame`` so results are easy to embed or export.

Functions
---------
plot_numerical_distributions  – Histograms + KDE for numeric columns.
plot_categorical_distributions – Bar charts for top-N categories.
plot_correlation_matrix        – Annotated heatmap of numeric correlations.
plot_scatter_premium_claims    – Scatter of TotalPremium vs TotalClaims
                                  coloured by a grouping variable.
plot_loss_ratio_by_group       – Horizontal bar chart of Loss Ratio per group.
plot_boxplots_outliers         – Box plots for outlier detection.
plot_temporal_trends           – Monthly claim frequency and severity trends.
plot_geographic_comparison     – Province-level comparison of a KPI.
summarise_loss_ratio           – Tabular Loss Ratio summary by group.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Suppress noisy warnings from older matplotlib / seaborn interactions.
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Style defaults – applied once when the module is imported.
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DPI = 120


# ---------------------------------------------------------------------------
# Numerical distributions
# ---------------------------------------------------------------------------

def plot_numerical_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    ncols: int = 3,
    bins: int = 50,
) -> plt.Figure:
    """
    Draw a histogram with a KDE overlay for each numeric column.

    This gives an immediate sense of skewness, modality, and the
    approximate range of values for financial features like
    ``TotalPremium`` and ``TotalClaims``.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    cols : list of str, optional
        Columns to plot.  Defaults to all ``float64`` / ``int64`` columns.
    ncols : int
        Number of subplot columns per row (default 3).
    bins : int
        Number of histogram bins (default 50).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    nrows = -(-len(cols) // ncols)          # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=FIGURE_DPI)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.hist(data, bins=bins, color="steelblue", edgecolor="white", alpha=0.7, density=True)
        # Overlay KDE for a smooth distributional shape.
        data.plot.kde(ax=ax, color="navy", lw=1.5)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Hide any unused subplot slots.
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Categorical distributions
# ---------------------------------------------------------------------------

def plot_categorical_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    top_n: int = 10,
    ncols: int = 2,
) -> plt.Figure:
    """
    Plot value-count bar charts for the most frequent categories.

    High-cardinality columns (e.g. ``PostalCode``) are truncated to the
    ``top_n`` most frequent values so the charts remain readable.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    cols : list of str, optional
        Categorical columns to plot.  Defaults to all ``category``/``object``
        dtype columns.
    top_n : int
        Maximum number of categories to display per column (default 10).
    ncols : int
        Number of subplot columns per row (default 2).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

    nrows = -(-len(cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), dpi=FIGURE_DPI)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        counts = df[col].value_counts().head(top_n)
        counts.plot.bar(ax=ax, color="teal", edgecolor="white")
        ax.set_title(f"{col} (top {top_n})", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    method: str = "pearson",
) -> plt.Figure:
    """
    Display an annotated heatmap of pairwise feature correlations.

    Useful for identifying multicollinearity before modeling and for
    understanding which features move together with claim or premium values.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    cols : list of str, optional
        Numeric columns to include.  Defaults to all numeric columns.
    method : str
        Correlation method: ``"pearson"``, ``"spearman"``, or ``"kendall"``
        (default ``"pearson"``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle (redundant)

    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) - 1)), dpi=FIGURE_DPI)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Correlation Matrix ({method.capitalize()})", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scatter: TotalPremium vs TotalClaims
# ---------------------------------------------------------------------------

def plot_scatter_premium_claims(
    df: pd.DataFrame,
    hue_col: str = "Province",
    sample_n: int = 5000,
    alpha: float = 0.4,
) -> plt.Figure:
    """
    Scatter plot of TotalPremium (x-axis) versus TotalClaims (y-axis).

    Points are coloured by ``hue_col`` (e.g. ``Province`` or ``PostalCode``)
    to reveal whether geographic or demographic segments explain the
    spread in claim amounts relative to the premium collected.

    A 45-degree reference line (break-even) is drawn so policies above it
    are unprofitable (claims exceed premium).

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (must contain ``TotalPremium`` and ``TotalClaims``).
    hue_col : str
        Column used to colour the points (default ``"Province"``).
    sample_n : int
        Maximum rows to plot; a random sample is drawn when the DataFrame
        exceeds this limit to keep rendering fast (default 5 000).
    alpha : float
        Point transparency (default 0.4).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = df[["TotalPremium", "TotalClaims", hue_col]].dropna()
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)
    sns.scatterplot(
        data=plot_df,
        x="TotalPremium",
        y="TotalClaims",
        hue=hue_col,
        alpha=alpha,
        s=20,
        ax=ax,
    )

    # Break-even line: claims == premium.
    lim = max(plot_df["TotalPremium"].max(), plot_df["TotalClaims"].max())
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", lw=1.2, label="Break-even")

    ax.set_title(f"TotalPremium vs TotalClaims by {hue_col}", fontweight="bold")
    ax.set_xlabel("Total Premium (ZAR)")
    ax.set_ylabel("Total Claims (ZAR)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Loss Ratio by group
# ---------------------------------------------------------------------------

def plot_loss_ratio_by_group(
    df: pd.DataFrame,
    group_col: str,
    top_n: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart of mean Loss Ratio for each category in ``group_col``.

    A vertical dashed line at Loss Ratio = 1.0 marks the break-even point.
    Segments to the right of this line are unprofitable on average.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (must contain ``LossRatio`` and ``group_col``).
    group_col : str
        Categorical column to group by (e.g. ``"Province"``, ``"VehicleType"``).
    top_n : int
        Show only the ``top_n`` groups ranked by Loss Ratio (default 15).

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = (
        df.groupby(group_col, observed=True)["LossRatio"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)), dpi=FIGURE_DPI)
    colors = ["#d62728" if v > 1.0 else "#2ca02c" for v in summary.values]
    summary.plot.barh(ax=ax, color=colors, edgecolor="white")

    ax.axvline(1.0, color="black", linestyle="--", lw=1.2, label="Break-even (LR=1)")
    ax.set_title(f"Average Loss Ratio by {group_col}", fontweight="bold")
    ax.set_xlabel("Mean Loss Ratio")
    ax.set_ylabel(group_col)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Outlier detection via box plots
# ---------------------------------------------------------------------------

def plot_boxplots_outliers(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    ncols: int = 3,
) -> plt.Figure:
    """
    Box plots for key numeric features to identify outliers.

    The IQR-based whiskers in a standard box plot highlight values
    that deviate strongly from the bulk of the distribution.  Outlier
    policies with extreme ``TotalClaims`` or ``CustomValueEstimate`` can
    distort regression models if left untreated.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    cols : list of str, optional
        Numeric columns to inspect.  Defaults to a curated set of
        financially relevant columns.
    ncols : int
        Subplot columns per row (default 3).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = [
            c for c in [
                "TotalPremium", "TotalClaims", "SumInsured",
                "CustomValueEstimate", "CalculatedPremiumPerTerm",
                "LossRatio", "Margin",
            ]
            if c in df.columns
        ]

    nrows = -(-len(cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=FIGURE_DPI)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", color="navy"),
                   medianprops=dict(color="red", lw=2),
                   flierprops=dict(marker=".", markersize=3, alpha=0.4))
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel(col)
        ax.set_xticks([])

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Outlier Detection – Box Plots", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Temporal trends
# ---------------------------------------------------------------------------

def plot_temporal_trends(df: pd.DataFrame) -> plt.Figure:
    """
    Line chart of monthly claim frequency and average claim severity.

    Claim **frequency** = fraction of policies with ``TotalClaims > 0``.
    Claim **severity** = mean ``TotalClaims`` among policies with a claim.

    A dual-axis design is used so both metrics (which have very different
    scales) are readable on the same chart.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (must contain ``TransactionMonth`` and
        ``TotalClaims``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    required = {"TransactionMonth", "TotalClaims"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    monthly = (
        df.groupby(df["TransactionMonth"].dt.to_period("M"))
        .agg(
            claim_frequency=("TotalClaims", lambda x: (x > 0).mean()),
            claim_severity=("TotalClaims", lambda x: x[x > 0].mean()),
        )
        .reset_index()
    )
    monthly["TransactionMonth"] = monthly["TransactionMonth"].astype(str)

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=FIGURE_DPI)
    color_freq = "steelblue"
    color_sev = "darkorange"

    ax1.plot(monthly["TransactionMonth"], monthly["claim_frequency"],
             marker="o", color=color_freq, lw=2, label="Claim Frequency")
    ax1.set_ylabel("Claim Frequency (proportion)", color=color_freq)
    ax1.tick_params(axis="y", labelcolor=color_freq)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(monthly["TransactionMonth"], monthly["claim_severity"],
             marker="s", color=color_sev, lw=2, linestyle="--", label="Claim Severity")
    ax2.set_ylabel("Avg Claim Severity (ZAR)", color=color_sev)
    ax2.tick_params(axis="y", labelcolor=color_sev)

    # Combine legends from both axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Monthly Claim Frequency and Severity (Feb 2014 – Aug 2015)",
                  fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Geographic comparison
# ---------------------------------------------------------------------------

def plot_geographic_comparison(
    df: pd.DataFrame,
    kpi_col: str = "LossRatio",
    province_col: str = "Province",
) -> plt.Figure:
    """
    Grouped bar chart comparing a KPI across South African provinces.

    Overlays a horizontal reference line at the portfolio-wide mean to
    make above/below-average provinces immediately apparent.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    kpi_col : str
        Numeric column to aggregate (default ``"LossRatio"``).
    province_col : str
        Column containing province labels (default ``"Province"``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = (
        df.groupby(province_col, observed=True)[kpi_col]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .reset_index()
    )

    portfolio_mean = df[kpi_col].mean()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIGURE_DPI)
    bars = ax.bar(summary[province_col], summary[kpi_col],
                  color="cornflowerblue", edgecolor="white")

    # Colour bars above/below the portfolio mean differently.
    for bar, val in zip(bars, summary[kpi_col]):
        bar.set_color("#d62728" if val > portfolio_mean else "#2ca02c")

    ax.axhline(portfolio_mean, color="navy", linestyle="--", lw=1.4,
               label=f"Portfolio mean ({portfolio_mean:.2f})")
    ax.set_title(f"{kpi_col} by Province", fontweight="bold")
    ax.set_xlabel("Province")
    ax.set_ylabel(f"Mean {kpi_col}")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tabular summary helper
# ---------------------------------------------------------------------------

def summarise_loss_ratio(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Return a tabular summary of Loss Ratio and Margin statistics by group.

    The table reports, for each category in ``group_col``:
    - Policy count.
    - Mean, median, and standard deviation of ``LossRatio``.
    - Total ``TotalPremium`` and ``TotalClaims`` collected.
    - Derived portfolio-level ``LossRatio`` (aggregate claims / aggregate premium).

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (must contain ``LossRatio``, ``TotalPremium``,
        and ``TotalClaims``).
    group_col : str
        Column to group by (e.g. ``"Province"``, ``"Gender"``).

    Returns
    -------
    pd.DataFrame
        Summary table sorted by portfolio-level ``LossRatio`` descending.
    """
    summary = df.groupby(group_col, observed=True).agg(
        policy_count=("TotalPremium", "count"),
        mean_loss_ratio=("LossRatio", "mean"),
        median_loss_ratio=("LossRatio", "median"),
        std_loss_ratio=("LossRatio", "std"),
        total_premium=("TotalPremium", "sum"),
        total_claims=("TotalClaims", "sum"),
    ).reset_index()

    # Portfolio-level loss ratio is more robust than averaging row-level ratios.
    summary["portfolio_loss_ratio"] = (
        summary["total_claims"] / summary["total_premium"].replace(0, np.nan)
    )

    return summary.sort_values("portfolio_loss_ratio", ascending=False).reset_index(drop=True)
