import numpy as np
import pandas as pd
from scipy.stats import kruskal


def anova_eta_squared(*groups):
    grand_mean = pd.concat(groups).mean()
    ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
    ss_total = sum((x - grand_mean) ** 2 for group in groups for x in group)
    eta_squared = ss_between / ss_total
    return eta_squared


def anova_cohen_f(*groups):
    eta_squared = anova_eta_squared(*groups)
    cohen_f = np.sqrt(eta_squared / (1 - eta_squared))
    return cohen_f


def kruskal_wallis_eta_squared(*groups):
    h_statistic, p_value = kruskal(*groups)
    k = len(groups)
    N = sum([len(group) for group in groups])
    return (h_statistic - k + 1) / (N - k)
