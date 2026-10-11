"""Statistical diagnostics for amount distributions and chronological cohorts."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    required = {'Class', 'Amount', 'Time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    data = df.sort_values('Time', kind='stable')
    fraud_amount = data.loc[data.Class == 1, 'Amount'].to_numpy()
    legit_amount = data.loc[data.Class == 0, 'Amount'].to_numpy()
    if len(fraud_amount) == 0 or len(legit_amount) == 0:
        raise ValueError('Both positive and negative cases are required for the Amount hypothesis test')
    mw = mannwhitneyu(fraud_amount, legit_amount, alternative='two-sided')
    n = len(data); split = n // 2
    first, second = data.iloc[:split], data.iloc[split:]
    counts = [int(first.Class.sum()), int(second.Class.sum())]
    totals = [len(first), len(second)]
    zstat, pval = proportions_ztest(counts, totals)
    ci_a = proportion_confint(counts[0], totals[0], method='wilson')
    ci_b = proportion_confint(counts[1], totals[1], method='wilson')
    return pd.DataFrame([
      {'hypothesis': 'Fraud and legitimate amount distributions differ', 'test': 'Mann-Whitney U, two-sided', 'statistic': float(mw.statistic), 'p_value': float(mw.pvalue), 'median_fraud_amount': float(np.median(fraud_amount)), 'median_legitimate_amount': float(np.median(legit_amount))},
      {'hypothesis': 'Fraud prevalence differs between first and second chronological halves', 'test': 'Two-proportion z-test', 'statistic': float(zstat), 'p_value': float(pval), 'first_half_rate': counts[0] / totals[0], 'first_half_ci_low': ci_a[0], 'first_half_ci_high': ci_a[1], 'second_half_rate': counts[1] / totals[1], 'second_half_ci_low': ci_b[0], 'second_half_ci_high': ci_b[1]},
    ])
