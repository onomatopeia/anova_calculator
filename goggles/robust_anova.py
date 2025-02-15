import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri

import logging

logger = logging.getLogger('colour')

def _one_way_anova(formula, data, iterations=3000):
    r_code = """
    library(WRS2)
    f <- function(formula, data, iter){
        return(t1way(formula = formula, data = data, tr=0, alpha=0.05, nboot = iter))
    }
    g <- function(formula, data){
        return(lincon(formula=formula, data=data, tr=0, alpha=0.05, method="hochberg"))
    }
    """
    robjects.r(r_code)
    func = robjects.r['f']
    results = func(formula, data, iterations)
    logger.debug(results)

    func = robjects.r['g']
    results = func(formula, data)
    logger.debug(results)



def one_way_anova(samples):
    df = pd.concat([
        pd.DataFrame(data = {
            'group': data_name,
            'values': values
        })
        for data_name, values in samples.items()
    ], ignore_index=True)
    with (robjects.default_converter + pandas2ri.converter).context():
        r_df = robjects.conversion.get_conversion().py2rpy(df)
    _one_way_anova('values ~ group', r_df)