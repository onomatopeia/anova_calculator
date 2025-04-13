import logging

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal, mannwhitneyu
import pingouin as pg
from goggles.stats import TestResult, interpret_p_values

logger = logging.getLogger("colour")


def mean_equality_between_groups(
    samples: dict[str, pd.Series],
    alpha: float = 0.05,
    marginal_alpha: float = 0.1
) -> bool:
    groups_names = []
    values = []
    for sample_name, sample_values in samples.items():
        groups_names.extend([sample_name] * len(sample_values))
        values.extend(sample_values.to_list())
    df = pd.DataFrame({'group': groups_names, 'value': values})

    welch_res = pg.welch_anova(dv='value', between='group', data=df).iloc[0]

    logger.debug('\nWelch\'s ANOVA')
    p_value = welch_res['p-unc']
    if p_value <= marginal_alpha:
        if p_value <= alpha:
            logger.debug(
                "Reject the null hypothesis: Some of the groups' averages consider to be not equal."
            )
        else:
            logger.debug(
                "Some of the groups' averages consider to be marginally not equal."
            )
        gh_res = pg.pairwise_gameshowell(dv='value', between='group', data=df)
        logger.debug('Games-Howell pairwise comparison')
        logger.debug(gh_res.to_markdown())
    else:
        logger.debug(
            f"Fail to reject the null hypothesis: The average of all groups assumed to be equal."
        )
    logger.debug(f"F Statistic: {welch_res['F']:.4f}, P-value: {p_value:.4f}")
    logger.debug(f"Observed effect size {welch_res['np2']}")
    return welch_res['p-unc'] <= marginal_alpha


def kruskal_wallis_nonparametric_anova(*groups, alpha=0.05, marginal_alpha=0.1) -> bool:
    res = TestResult._make(kruskal(*groups))
    logger.debug('\nKruskal-Wallis Nonparametric Test for Equality of Means')
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
    logger.debug(f"H Statistic: {res.statistic:.4f}, P-value: {res.pvalue:.4f}")
    return res.pvalue <= marginal_alpha


def pairwise_comparisons_dunn(samples, alpha: float = 0.05, correction='holm') -> bool:
    """Post hoc pairwise test for multiple comparisons of mean rank sums (Dunn’s test).

    :param samples: A factor value - observations dictionary.
    :param alpha: Significance level, defaults to 0.05.
    :param correction: Defaults to Holm-Bonferroni correction as it provides a good balance
    between reducing false positives and maintaining statistical power. Other corrections can be
    e.g. 'bonferroni' for Bonferroni method or 'fdr_bh' for Benjamini-Hochberg.
    :return: Whether at least one pair is significant
    """
    res: pd.DataFrame = sp.posthoc_dunn(
        [list(values) for values in samples.values()],
        p_adjust=correction
    )
    res.columns = samples.keys()
    res.index = samples.keys()
    mask = np.triu(np.ones(res.shape), k=1).astype(bool)
    df_transformed = res.where(mask).stack().reset_index()
    df_transformed.columns = ['Group 1', 'Group 2', 'p-value']
    df_transformed['Significant'] = interpret_p_values(df_transformed['p-value'], alpha)
    logger.debug('\nPost-hoc Dunn\'s Test for multiple comparisons of mean rank sums')
    logger.debug(df_transformed)
    return (df_transformed['p-value'] <= alpha).any()


def mann_whitney_u_test(sample1, sample2, alpha: float = 0.05) -> bool:
    res = mannwhitneyu(sample1, sample2)
    logger.debug(
        f"Two independent samples Mann-Whitney U-test: U = {res.statistic}, p = {res.pvalue}"
    )
    return res.pvalue <= alpha