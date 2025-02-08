import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from goggles import assumptions, descriptive, nonparametric, parametric
from goggles.power import calculate_anova_power

logger = logging.getLogger("colour")


def analysis_of_variance(factor, samples, output_folder, col):
    logger.debug('Descriptive statistics')
    descriptive.describe(samples)
    logger.debug('ANOVA Assumptions')
    assumptions_passed = True
    logger.debug('\n1. Equal cell sizes')
    equal_cell_sizes = assumptions.equal_size_samples(*samples.values())
    assumptions_passed &= equal_cell_sizes
    logger.debug('\n2. Normality')
    assumptions_passed &= assumptions.normality(samples, output_folder)
    logger.debug('\n3. Homoscedasticity')
    assumptions_passed &= assumptions.equal_variances(*samples.values())
    logger.debug('\nKruskal-Wallis Test Assumptions')
    logger.debug('4. Similarity of shape')
    logger.debug('Shape should be verified manually in the associated distribution plots.')
    assumptions.similarity_of_shape(factor, col, samples, output_folder)

    if assumptions_passed:
        logger.debug('\nAll ANOVA assumptions have passed.')
        if parametric.mean_equality_between_groups(*samples.values()):
            result = parametric.pairwise_comparisons(samples)
            if result:
                logger.debug('At least one pair has significantly different means by ANOVA.')
            else:
                logger.debug('No significant differences in pairs\' means by ANOVA.')
    else:
        logger.debug('\nNot all ANOVA assumptions have passed. Switching to Welch\'s ANOVA.')
        nonparametric.mean_equality_between_groups(samples)


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


def evaluate_differences_in_means(
    data_file_path: Path,
    columns: Sequence[str],
    factor: str,
    sheets: dict[str, str],
) -> None:
    dfs = read_data(data_file_path, sheets)
    results_folder = Path(__file__).parents[1].joinpath('results', factor)

    for col in columns:
        logger.debug(f"Variable: {col}")
        output_folder = results_folder.joinpath(col)
        output_folder.mkdir(exist_ok=True, parents=True)

        for df_name, df in dfs.items():
            if factor == 'TTFF' and 'bucket' in col:
                dfs[df_name].loc[df[col] <= 11] = np.nan

        #values, lambda_ = scipy.stats.boxcox(pd.concat([df[col].dropna() for df in dfs.values()]))
        #logger.debug(f'{col} lambda {lambda_}')
        lambda_ = -2
        samples = {
            df_name: pd.Series(scipy.stats.boxcox(df[col].dropna(), lambda_))
            for df_name, df in dfs.items()
        }

        analysis_of_variance(factor, samples, output_folder, col)
        logger.debug("--------------------------------------------------------------------------\n")
