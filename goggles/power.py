from statsmodels.stats.power import FTestAnovaPower


def calculate_anova_power(effect_size, *groups):
    alpha = 0.05
    num_groups = len(groups)
    nobs = sum([len(group) for group in groups])
    anova_power = FTestAnovaPower()
    return anova_power.solve_power(
        effect_size=effect_size,
        nobs=nobs,
        alpha=alpha,
        power=None,
        k_groups=num_groups,
    )
