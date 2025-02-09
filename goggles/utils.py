import numpy as np
import pandas as pd


def trim_data(data: pd.Series, trim_fraction: float = 0.1):
    trim_one_side = 100 * trim_fraction / 2
    lower_limit = np.percentile(data, trim_one_side)
    upper_limit = np.percentile(data, 100 - trim_one_side)
    output = pd.Series([x for x in data if lower_limit <= x <= upper_limit], name=data.name)
    return output
