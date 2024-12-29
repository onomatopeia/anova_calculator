import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.colors
import plotly.express as px
from scipy.stats import chisquare, levene, probplot, shapiro

from goggles.stats import TestResult


def equal_size_samples(*groups, alpha=0.05) -> bool:
    observed = [len(group) for group in groups]
    num_groups = len(groups)
    total_count = sum(observed)
    expected = [total_count / num_groups] * num_groups

    res = TestResult._make(chisquare(f_obs=observed, f_exp=expected))
    print('Chi-Squared Test for Equal Samples\' Sizes')
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
    print('Shapiro-Wilk Test for Normality')
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
    print('Levene Test for Homoscedasticity')
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


def similarity_of_shape(
    factor: str,
    variable_name: str,
    groups: dict[str, pd.Series],
    output_folder: Path
) -> None:
    goggles = 'Goggles'
    data = {
        factor: pd.concat(groups.values()),
        goggles: list(
            itertools.chain.from_iterable(
                [g_name] * len(group) for g_name, group in groups.items()
            )
        )
    }
    df = pd.DataFrame(data)
    colors = {
        'Transparent': plotly.colors.qualitative.Plotly[0],
        'Yellow': plotly.colors.qualitative.Plotly[9],
        'Red': plotly.colors.qualitative.Plotly[1],
    }
    fig = px.histogram(df, x=goggles, color=factor, marginal='box', color_discrete_map=colors)
    fig.update_layout(template='plotly_white')
    fig.write_image(output_folder.joinpath(f"distplot_{variable_name}.png"), scale=3)