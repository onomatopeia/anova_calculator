from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway, chisquare, shapiro, probplot, levene

TestResult = namedtuple('TestResult', ('statistic', 'pvalue'))


def one_way_anova(*groups, alpha: float = 0.05) -> bool:
    res = TestResult._make(f_oneway(*groups))
    if res.pvalue <= alpha:
        print("Reject the null hypothesis: Some of the groups' averages consider to be not equal.")
    else:
        print(
            f"Fail to reject the null hypothesis: The average of all groups assumed to be equal."
        )
    print(f"F Statistic: {res.statistic:.4f}, P-value: {res.pvalue:.4f}")
    return res.pvalue <= alpha


def equal_size_samples(*groups, alpha=0.05) -> bool:
    observed = [len(group) for group in groups]
    num_groups = len(groups)
    total_count = sum(observed)
    expected = [total_count / num_groups] * num_groups

    res = TestResult._make(chisquare(f_obs=observed, f_exp=expected))
    if res.pvalue <= alpha:
        print(f"Reject the null hypothesis: The counts {observed} are not evenly distributed.")
    else:
        print(
            f"Fail to reject the null hypothesis: "
            f"The counts {observed} are roughly evenly distributed."
        )
    print(f"Chi-squared Statistic: {res.statistic:.4f}, P-value: {res.pvalue:.4f}")
    return res.pvalue > alpha


def normality(
    groups: dict[str, pd.Series],
    output_folder: Path,
    alpha: float = 0.05,
) -> bool:
    results = {}
    normality_pass = True
    for g_name, group in groups.items():
        results[g_name] = res = TestResult._make(shapiro(group))
        if res.pvalue < alpha:
            print(
                f"Reject the null hypothesis: The sample {g_name} is not normally distributed."
            )
            if len(group) >= 30:
                print(
                    'Sample size is equal or greater than 30 and can be considered sufficient for '
                    'CLT to hold.'
                )
            else:
                normality_pass = False
        else:
            print(
                f"Fail to reject the null hypothesis: "
                f"The sample {g_name} is roughly normally distributed."
            )
        print(f"{g_name}: W Statistic: {res.statistic:.4f}, P-value: {res.pvalue:.4f}")

        probplot(group, dist="norm", plot=plt)
        plt.title(f'Probability Plot - {g_name} Goggles')
        plt.savefig(output_folder.joinpath(f'{g_name}.png'), bbox_inches='tight', dpi=300)
        plt.close()

    return normality_pass


def equal_variances(*groups: pd.Series, alpha: float = 0.05) -> bool:
    res = TestResult._make(levene(*groups))
    if res.pvalue <= alpha:
        print(
            f"Reject the null hypothesis: Samples' variances are not equal."
        )
    else:
        print(
            f"Fail to reject the null hypothesis: Samples' variances are roughly equal."
        )
    print(f"Test Statistic: {res.statistic:.4f}, P-value: {res.pvalue:.4f}")
    return res.pvalue > alpha


def tukey_hsd(samples: dict[str, pd.Series]) -> None:
    return None
