import logging
from collections.abc import Sequence

import pandas as pd
from scipy.stats import f_oneway, tukey_hsd, ttest_ind
from scipy.stats._hypotests import TukeyHSDResult

from goggles.stats import TestResult, interpret_p_values

logger = logging.getLogger("colour")


def mean_equality_between_groups(*groups, alpha: float = 0.05, marginal_alpha: float = 0.1) -> bool:
    res = TestResult._make(f_oneway(*groups))
    logger.debug('\nANOVA Test for Equality of Means')
    if res.pvalue <= marginal_alpha:
        if res.pvalue <= alpha:
            logger.debug(
                "Reject the null hypothesis: Some of the groups' averages consider to be not equal."
            )
        else:
            logger.debug(
                "Some of the groups' averages consider to be marginally not equal."
            )
    else:
        logger.debug(
            f"Fail to reject the null hypothesis: The average of all groups assumed to be equal."
        )
    logger.debug(f"F Statistic: {res.statistic:.4f}, P-value: {res.pvalue:.4f}")
    return res.pvalue <= marginal_alpha


def _tukey_hsd_results_info(
    names: Sequence[str],
    res: TukeyHSDResult,
    alpha: float = 0.05,
) -> pd.DataFrame:
    confidence_level = res.confidence_interval(confidence_level=1 - alpha)

    rows = []

    for i in range(res.pvalue.shape[0]):
        for j in range(i + 1, res.pvalue.shape[0]):
            rows.append(
                (
                    names[i],
                    names[j],
                    res.statistic[i, j],
                    res.pvalue[i, j],
                    confidence_level.low[i, j],
                    confidence_level.high[i, j],
                )
            )
    result = pd.DataFrame.from_records(
        rows,
        columns=['Group 1', 'Group 2', 'Statistic', 'p-value', 'Lower CI', 'Upper CI']
    )
    result['Significant'] = interpret_p_values(result['p-value'], alpha)

    return result


def pairwise_comparisons(samples: dict[str, pd.Series], alpha: float = 0.05) -> bool:
    keys = list(samples.keys())
    res = tukey_hsd(*samples.values())
    logger.debug(
        f"Tukey's HSD Pairwise Group Comparisons at {(1 - alpha) * 100:.1f}% Confidence Interval)\n"
    )
    res_df = _tukey_hsd_results_info(keys, res, alpha)
    logger.debug(res_df)

    return any(res.pvalue.ravel() <= alpha)


def paired_t_test(
    sample1: pd.Series,
    sample2: pd.Series,
    equal_var: bool,
    nan_policy: str = 'omit',
    alpha: float = 0.05,
) -> bool:
    res = ttest_ind(sample1, sample2, equal_var=equal_var, nan_policy=nan_policy)
    if equal_var:
        logger.debug(
            f"Two independent samples standard t-test: t = {res.statistic}, p = {res.pvalue}."
        )
    else:
        logger.debug(
            f"Two independent samples Welch's t-test: t = {res.statistic}, p = {res.pvalue}"
        )
    return res.pvalue <= alpha
