import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import median_test

from goggles import assumptions, descriptive, effect_size, nonparametric, parametric
from goggles.power import calculate_anova_power
from goggles.utils import trim_data

logger = logging.getLogger("colour")


def bootstrap_anova(groups, n_bootstrap=10000):
    observed_means = [np.mean(group) for group in groups]
    observed_stat = np.var(observed_means)

    combined = np.concatenate(groups)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        resampled = [np.random.choice(combined, size=len(group), replace=True) for group in groups]
        bootstrap_means = [np.mean(resampled_group) for resampled_group in resampled]
        bootstrap_stats.append(np.var(bootstrap_means))

    p_value = np.mean(np.array(bootstrap_stats) >= observed_stat)
    return observed_stat, p_value


def analysis_of_variance(factor, samples, output_folder, col):
    logger.debug('Descriptive statistics')
    descriptive.describe(samples)
    logger.debug('ANOVA Assumptions')
    assumptions_passed = True
    logger.debug('\n1. Equal cell sizes')
    equal_cell_sizes = assumptions.equal_size_samples(*samples.values())
    assumptions_passed &= equal_cell_sizes
    logger.debug('\n2. Normality')
    normality = assumptions.normality(samples, output_folder)
    assumptions_passed &= normality
    logger.debug('\n3. Homoscedasticity')
    homoscedasticity = assumptions.equal_variances(*samples.values())
    assumptions_passed &= homoscedasticity
    logger.debug('\nKruskal-Wallis Test Assumptions')
    logger.debug('4. Similarity of shape')
    logger.debug('Shape should be verified manually in the associated distribution plots.')
    assumptions.similarity_of_shape(factor, col, samples, output_folder)

    stat, p_value, _, _ = median_test(*samples.values())
    logger.debug(f"\nMood's Median Test statistic: {stat}, P-value: {p_value}")

    stat, p_val = bootstrap_anova([*samples.values()])
    logger.debug(f"\nBootstrap ANOVA statistic: {stat}, P-value: {p_val}")

    if assumptions_passed:
        logger.debug('\nAll ANOVA assumptions have passed.')
        if parametric.mean_equality_between_groups(*samples.values()):
            result = parametric.pairwise_comparisons(samples)
            if result:
                logger.debug('At least one pair has significantly different means by ANOVA.')
            else:
                logger.debug('No significant differences in pairs\' means by ANOVA.')
        logger.debug(f'Effect size: {effect_size.anova_eta_squared(*samples.values())}')
    elif not homoscedasticity:
        logger.debug('\nNot all ANOVA assumptions have passed. Switching to Welch\'s ANOVA.')
        nonparametric.mean_equality_between_groups(samples)
    elif not normality:
        logger.debug(
            '\nNormality not passed, but variance roughly equal. Switching to Kruskal-Wallis test'
        )
        if nonparametric.kruskal_wallis_nonparametric_anova(*samples.values()):
            nonparametric.pairwise_comparisons_dunn(samples)
            logger.debug(f'Effect size: {effect_size.kruskal_wallis_eta_squared(*samples.values())}')




def read_data(file_path: Path, sheets: dict[str, str]):
    return {
        group_name: pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            usecols=range(1, 8),
            engine='odf',
            skiprows=1,
        )
        for group_name, sheet_name in sheets.items()
    }


def calculate_boxcox_lambdas(dfs, columns, factor):
    lambdas_ = []
    for col in columns:
        samples = collate_samples(dfs, col, factor)
        all_samples = pd.concat(samples.values())
        _, lambda_ = scipy.stats.yeojohnson(all_samples)
        lambdas_.append(lambda_)
    logger.debug(f'Calculated lambdas: {lambdas_}')
    return lambdas_


def collate_samples(dfs, col, factor):
    for df_name, df in dfs.items():
        if factor == 'TTFF' and 'bucket' in col:
            dfs[df_name].loc[df[col] >= 11] = np.nan

    samples = {
        df_name: df[col].dropna()
        for df_name, df in dfs.items()
    }
    return samples


def evaluate_differences_in_means(
    data_file_path: Path,
    columns: Sequence[str],
    factor: str,
    sheets: dict[str, str],
    lambda_: float | list[float] = 1,
    trim_fraction: float = 0.0,
    calculate_boxcox: bool = False,
) -> None:
    dfs = read_data(data_file_path, sheets)
    results_folder = Path(__file__).parents[1].joinpath('results', factor)

    if not isinstance(lambda_, float):
        lambda_ = [lambda_] * len(columns)
    if calculate_boxcox:
        lambda_ = calculate_boxcox_lambdas(dfs, columns, factor)

    for col, col_lambda in zip(columns, lambda_):
        logger.debug(f"Variable: {col}")
        output_folder = results_folder.joinpath(col)
        output_folder.mkdir(exist_ok=True, parents=True)

        samples = collate_samples(dfs, col, factor)

        if 0 < trim_fraction < 1:
            logger.debug(
                f"Trimming data to [{trim_fraction / 2}, {1 - trim_fraction / 2}] quantiles"
            )
            samples = {
                sample_name: trim_data(sample_data, trim_fraction)
                for sample_name, sample_data in samples.items()
            }
        else:
            logger.debug(f"Trim fraction {trim_fraction} outside of (0,1), not trimming")

        logger.debug(f"Performing Yeo-Johnson transformation with lambda={col_lambda}")
        samples = {
            sample_name: pd.Series(
                scipy.stats.yeojohnson(sample_data, col_lambda),
                name=sample_data.name,
            )
            for sample_name, sample_data in samples.items()
        }

        analysis_of_variance(factor, samples, output_folder, col)
        logger.debug("--------------------------------------------------------------------------\n")
