import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from goggles import assumptions, nonparametric, parametric

logger = logging.getLogger("colour")


def analysis_of_variance(factor, samples, output_folder, col):
    logger.debug('ANOVA Assumptions')
    assumptions_passed = True
    logger.debug('\n1. Equal cell sizes')
    assumptions_passed &= assumptions.equal_size_samples(*samples.values())
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
        logger.debug('\nNot all ANOVA assumptions have passed. Switching to Kruskal-Wallis Test.')
        if nonparametric.mean_equality_between_groups(*samples.values()):
            result = nonparametric.pairwise_comparisons(samples)
            if result:
                logger.debug(
                    'At least one pair has significantly different means by Kruskal-Wallis test.'
                    )
            else:
                logger.debug('No significant differences in pairs\' means by Kruskal-Wallis test.')


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
        if factor == 'TTFF' and 'bucket' in col:
            samples = {
                df_name: df.loc[df[col] <= 11, col].dropna() for df_name, df in dfs.items()
            }
        else:
            samples = {df_name: df[col].dropna() for df_name, df in dfs.items()}
        analysis_of_variance(factor, samples, output_folder, col)
        logger.debug("--------------------------------------------------------------------------\n")
