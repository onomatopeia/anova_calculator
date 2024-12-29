from collections import namedtuple
from enum import StrEnum, auto

import pandas as pd

TestResult = namedtuple('TestResult', ('statistic', 'pvalue'))


class StatisticalSignificance(StrEnum):
    Yes = auto()
    Marginally = auto()
    No = auto()


def interpret_p_values(
    p_values: pd.Series,
    alpha: float = 0.05,
    marginal_significance: float = 0.1,
) -> pd.Series:
    result = pd.Series(data=StatisticalSignificance.No.value, index=p_values.index)
    result.loc[p_values <= marginal_significance] = StatisticalSignificance.Marginally.value
    result.loc[p_values <= alpha] = StatisticalSignificance.Yes.value
    return result
